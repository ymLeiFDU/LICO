import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.label_mapping import pdists
from utils.label_mapping import sim_matrix, sim_matrix_pre
from models.text_encoder import load_clip_to_cpu, TextEncoder, PromptLearner, PromptLearnerBERT
from models.clip import clip

from models.text_encoder import get_Cifar100_ClassNames

momentum = 0.001


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetPrompt(nn.Module):
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, is_remix=False, args = None):
        super(WideResNetPrompt, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]


        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.emb_temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        '''
        Prompt Learning
        '''
        classnames = ['plane', 'car', 'automobile', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # classnames = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

        # Text encoder
        self.clip_model = load_clip_to_cpu()
        self.text_encoder = TextEncoder(self.clip_model)
        self.prompt_learner = PromptLearner(classnames, self.clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # self.token_fc = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     # nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512)
        #     )

        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 64)
            )

        # self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.fc_768to512 = nn.Linear(768, 512)

    def forward(self, x, args, 
        targets = None, 
        w_distance = None,
        mode = 'train'):

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)  
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        feature_maps = out.view(out.shape[0], out.shape[1], -1)

        out = F.adaptive_avg_pool2d(out, 1)
        
        out = out.view(-1, self.channels)
        emb = out

        emb_temp = self.emb_temp
        emb_matrix = self._emb_SimMatrix(emb, temp = emb_temp, norm = True)
        

        text_features = []
        prompts = self.prompt_learner() # [100, 77, 512]
        text_features = self.text_encoder(prompts, self.tokenized_prompts) # [100, 512]


        
        if args.language and mode == 'train':
            out = self.fc(out)

            # prompt learning
            
            text_features_w = self.mlp(text_features)
            text_features_w = text_features_w.view(text_features_w.shape[0], -1) # (100, 64)
            text_features_w = text_features_w.expand(feature_maps.shape[0], text_features_w.shape[0], text_features_w.shape[1])
            
            feature_maps = F.normalize(feature_maps, dim = 2)
            text_features_w = F.normalize(text_features_w, dim = 2)
            with torch.no_grad():
                P, C = w_distance(feature_maps, text_features_w)
            w_loss = torch.sum(P * C, dim=(-2, -1)).mean()

            label_distribution, _ = sim_matrix_pre(
                targets, text_features, self.emb_temp, token_fc = None, noise = False)
            return out, emb_matrix, emb, w_loss, label_distribution
        if mode == 'test':
            out = self.fc(out)
            label_distribution, _ = sim_matrix_pre(
                targets, text_features, self.emb_temp, token_fc = None, noise = False)
            return out, emb_matrix, emb, label_distribution

    def _emb_SimMatrix(self, emb, temp, norm = True):

        if norm:
            emb = F.normalize(emb, dim = -1)
        else:
            pass

        dist = pdists(emb, noise = False)
        matrix = F.softmax(dist / temp, dim = 1)

        return matrix


class build_WideResNet:
    def __init__(self, first_stride=1, depth=28, widen_factor=2, bn_momentum=0.01, leaky_slope=0.0, dropRate=0.0,
                 use_embed=False, is_remix=False):
        self.first_stride = first_stride
        self.depth = depth
        self.widen_factor = widen_factor
        self.bn_momentum = bn_momentum
        self.dropRate = dropRate
        self.leaky_slope = leaky_slope
        self.use_embed = use_embed
        self.is_remix = is_remix

    def build(self, num_classes):
        return WideResNet(
            first_stride=self.first_stride,
            depth=self.depth,
            num_classes=num_classes,
            widen_factor=self.widen_factor,
            drop_rate=self.dropRate,
            is_remix=self.is_remix,
        )


if __name__ == '__main__':
    wrn_builder = build_WideResNet(1, 10, 2, 0.01, 0.1, 0.5)
    wrn = wrn_builder.build(10)
    print(wrn)
