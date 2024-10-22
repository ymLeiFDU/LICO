'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import pickle
import yaml
import random
import configs
import datetime
import os
import argparse

from models import *
from models.resnet_Prompt import ResNet50_Prompt

from models import wideresnet_Prompt
from models.modules.ema import EMA
from utils.utils import progress_bar, save_args, Tee
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from data.data_utils import split_ssl_data
from data.sample_cifar10 import SamCifarDataset

from models.modules.sinkhorn_distance import SinkhornDistance




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(args.gpu) > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, epoch, model, ema, lens, trainloader):
    model.train()
    train_loss = 0
    kl_loss = 0
    ce_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()

        outputs, emb_matrix, emb, w_loss, label_distribution = model(
                    inputs, args, targets, w_distance = w_distance, mode = 'train')
        if len(args.gpu) > 1:
            w_loss = torch.sum(w_loss)


        ce = criterion(outputs, targets)

        kl_loss = kl(emb_matrix.log(), label_distribution)


        loss = args.lambda_ce * ce + args.alpha * kl_loss + args.beta * w_loss
        loss.backward()
        optimizer.step()


        if args.use_ema:
            ema.update_params()

        train_loss += loss.item()
        ce_loss += ce.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    if args.use_ema:
        ema.update_buffer()

    return model, ce_loss


def test(args, epoch, model, ema, lens):
    global best_acc
    global best_epoch
    global best_model
    model.eval()
    test_loss = 0
    ce_loss = 0
    kl_loss = 0
    correct = 0
    total = 0

    if args.use_ema:
        ema.apply_shadow()
        ema.model.eval()
        ema.model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()


            outputs, emb_matrix, emb, label_distribution = ema.model(
                                            inputs, args, targets, mode = 'test')

            ce = criterion(outputs, targets)
            kl = kl(emb_matrix.log(), label_distribution)

            loss = ce

            test_loss += loss.item()
            ce_loss += ce.item()
            kl_loss += kl.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    if args.use_ema:
        ema.restore()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_dir + '/' + args.ablation + '.pth')
        
        best_acc = acc
        best_model = model
        best_epoch = epoch


    if epoch == args.epochs - 1:
        torch.save(best_model, checkpoint_dir + '/trained_models' + '/Acc_{:.4f}_epoch_{}_model.pth'.format(best_acc, best_epoch))
        torch.save(best_model.state_dict(), checkpoint_dir + '/trained_models' + '/Acc_{:.4f}_epoch_{}_model_dict.pth'.format(best_acc, best_epoch))
    
    return best_acc, best_epoch, kl_loss, ce_loss
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch-size', '-bs', default = 128, type=int)
    parser.add_argument('--test-batch-size', '-tbs', default = 128, type=int)
    parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
    parser.add_argument('--gpu', type = str, default = '0')
    parser.add_argument('--save-files', '-save', action = 'store_true')
    parser.add_argument('--model', type = str, default = 'resnet')
    parser.add_argument('--epochs', type = int, default = 200)
    parser.add_argument('--ablation', type = str, default = 'cifar10_CE_ResNet18')
    parser.add_argument('--use-ema', '-ema', action = 'store_true')
    parser.add_argument('--ema-alpha', type = float, default = 0.999)
    parser.add_argument('--dataset', type = str, default = 'cifar10')
    parser.add_argument('--num-labels', type = str, default = 'full')

    parser.add_argument('--lambda-ce', type = float, default = 1.)
    parser.add_argument('--alpha', type = float, default = 10)
    parser.add_argument('--beta', type = float, default = 1.)
    parser.add_argument('--shot', type = str, default = None)
    parser.add_argument('--max-iter-ot', type = int, default = 100)

    parser.add_argument('--seed', default=None, type=int, help="random seed")
    


    args = parser.parse_args()

    # set_seed(args)

    t = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    '''
    ==================== Saving files ====================
    '''
    with open('configs/paths.yaml', 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)

    if args.save_files:
        project_dir = './'
        if not os.path.exists(project_dir + '/ckpt/' + args.model):
            os.mkdir(project_dir + '/ckpt/' + args.model)

        checkpoint_dir = project_dir + '/ckpt/' + args.model + '/' + t + '_' + args.ablation

        print('Files saving dir: ', checkpoint_dir)
        files = cfgs['project_files']

        save_args(checkpoint_dir, files)
        logger = Tee(checkpoint_dir + '/log.txt', 'a')


        summary = TensorboardSummary(checkpoint_dir)
        writer = summary.create_summary()


    best_acc = 0  
    start_epoch = 0  

    '''
    ==================== Datasets ====================
    '''
    mean, std = {}, {}
    mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
    mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
    mean['svhn'] = [0.4380, 0.4440, 0.4730]
    mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
    mean['imagenet'] = [0.485, 0.456, 0.406]

    std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
    std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
    std['svhn'] = [0.1751, 0.1771, 0.1744]
    std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
    std['imagenet'] = [0.229, 0.224, 0.225]
    print('==> Preparing data..')
    if args.dataset in ['cifar10', 'aircraft', 'cars', 'flowers', 'cub']:
        norm_mean, norm_std = mean['cifar10'], std['cifar10']
    elif args.dataset == 'cifar100':
        norm_mean, norm_std = mean['cifar100'], std['cifar100']
    elif args.dataset == 'svhn':
        norm_mean, norm_std = mean['svhn'], std['svhn']

    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])



    
    if args.dataset == 'cifar10':
        wrn_width = 2
        trainset = torchvision.datasets.CIFAR10(
                root='/data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR10(
                root='/data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
                testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

        classes = ['plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = 10
        if args.num_labels == 'full':
            pass
        else:
            num_labels = int(args.num_labels)
            lb_data, lbs, ulb_data, ulbs = split_ssl_data(
                trainset.data, trainset.targets, num_labels, num_classes = 10)
            trainset = SamCifarDataset(lb_data, lbs, transform=transform_train)
            trainloader = DataLoader(
                                    trainset,
                                    batch_size = args.batch_size,
                                    num_workers = 2,
                                    shuffle=True)
        model = wideresnet_Prompt.WideResNetPrompt(
            first_stride=1, depth=28, widen_factor=wrn_width, 
            drop_rate=0.0, num_classes = num_classes, args = args)

    elif args.dataset == 'cifar100':
        wrn_width = 8
        trainset = torchvision.datasets.CIFAR100(
                root='./data/cifar100/', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(
                root='./data/cifar100/', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
                testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

        classes = trainset.classes
        num_classes = 100

        if args.num_labels == 'full':
            pass
        else:
            num_labels = int(args.num_labels)
            lb_data, lbs, ulb_data, ulbs = split_ssl_data(
                trainset.data, trainset.targets, num_labels, num_classes = 100)
            trainset = SamCifarDataset(lb_data, lbs, transform=transform_train)
            trainloader = DataLoader(
                                    trainset,
                                    batch_size = args.batch_size,
                                    num_workers = 2,
                                    shuffle=True)
        model = wideresnet_Prompt.WideResNetPrompt(
            first_stride=1, depth=28, widen_factor=wrn_width, 
            drop_rate=0.0, num_classes = num_classes, args = args)

    elif args.dataset == 'svhn':
        wrn_width = 2
        if args.num_labels == 'full':
            trainset = torchvision.datasets.SVHN(
                    root='./data/svhn/', split='train', download=True, transform=transform_train)
            classes = trainset.labels
            extraset = torchvision.datasets.SVHN(
                    root='./data/svhn/', split='extra', download=True, transform=transform_train)
            trainset += extraset
            trainloader = torch.utils.data.DataLoader(
                    trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            testset = torchvision.datasets.SVHN(
                    root='./data/svhn/', split='test', download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(
                    testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        else:
            trainset = SVHNNShotDataset(num_shots = n_shot, train_mode = 'train')
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            testset = SVHNNShotDataset(num_shots = n_shot, train_mode = 'test')
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
        model = wideresnet_Prompt.WideResNetPrompt(
            first_stride=1, depth=28, widen_factor=wrn_width, 
            drop_rate=0.0, num_classes = 10, args = args)
        classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        num_classes = 10





    '''
    ==================== Build Models ====================
    '''
    print('==> Building model..')
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu[0]))


    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpu.split(',')])
    model = model.cuda()



    '''
    ==================== Training Settings ====================
    '''
    if args.ema_alpha != 0:
        print('==> Training With EMA ...')
        ema = EMA(model, alpha = args.ema_alpha)
    else:
        print('==> Training NO EMA ...')

    # 
    global w_distance
    w_distance = SinkhornDistance(eps=0.1, max_iter=args.max_iter_ot, reduction='sum').cuda()
    global kl
    kl = nn.KLDivLoss(log_target=False, reduction = 'batchmean').cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)


    '''
    ==================== Training ====================
    '''

    for epoch in range(start_epoch, start_epoch+args.epochs):
        print('\n==> Epoch: %d / %d, Ablation: [ %s, %s ], %s' % (epoch, args.epochs, args.ablation, args.model, t))

        if args.use_ema:
            ema = ema
        else:
            ema = None

        model, train_kl, train_ce = train(args, epoch, model, 
            ema = ema, 
            lens = trainset.__len__()/args.batch_size,
            trainloader = trainloader)


        best_acc, best_epoch, test_kl, test_ce = test(
            args, epoch, model, ema = ema, lens = testset.__len__()/args.batch_size)
        scheduler.step()


    print('Training done, best Acc: {:2f}, @ epoch {:d}'.format(best_acc, best_epoch))











