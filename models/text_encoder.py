import os.path as osp
from collections import OrderedDict
import math

import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import json

from transformers import AlignTextModel, AlignProcessor, AlignModel

_tokenizer = _Tokenizer()


def load_clip_to_cpu():
    backbone_name = "ViT-B/32"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root = os.path.expanduser("/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def get_ImageNet_ClassNames():
    text_file = './imagenet_class_index.json'
    with open(text_file, 'r', encoding = 'utf-8') as f:
        l = json.load(f)
    names = []
    for i in range(len(l)):
        names.append(l[str(i)][-1])
    return names

def get_Cifar100_ClassNames():
    text_file = './cifar100_names.json'
    with open(text_file, 'r', encoding = 'utf-8') as f:
        l = json.load(f)
    names = []
    for i in range(len(l)):
        names.append(l[str(i)])
    return names

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        clip_model = clip_model.type(torch.FloatTensor)
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # with torch.no_grad():
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x



class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 12
        ctx_init = None
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.N = 1

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if False:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
                print('ctx_vectors: ', ctx_vectors.shape)
            
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
            ctx_vectors = torch.cat([torch.roll(ctx_vectors, shifts=1, dims=1) for i in range(n_ctx)], dim = 0)
            print('ctx_vectors: ', ctx_vectors.shape)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]   
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(prompts)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) 
        print(tokenized_prompts.shape)
        tokenized_prompts = tokenized_prompts.repeat(n_ctx, 1) 
        print('token prompts:', tokenized_prompts.shape)


        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 
        print('tokenized prompts:', embedding.shape, 'ctx: ', self.ctx.shape)


        self.register_buffer("token_prefix", embedding[:, :1, :]) 
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :]) 

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = 'end'

    def _ctx_shuffle(self, prefix, suffix, ctx, cls_loc = 'end', shuffleCLS = False):

        rand_idx = torch.randperm(ctx.shape[1])
        shuffled_ctx = ctx[:, rand_idx, :]
        return shuffled_ctx


    def forward(self):
        
        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1)
        ctx = ctx.permute(1, 0, 2, 3)

        ctx = ctx.contiguous().view(self.n_ctx*self.n_cls, self.n_ctx, ctx.shape[3])
        prefix = self.token_prefix
        suffix = self.token_suffix


        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  
                    ctx,     
                    suffix,  
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     
                        ctx_i_half1,  
                        class_i,      
                        ctx_i_half2,  
                        suffix_i,     
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  
                        class_i,   
                        ctx_i,     
                        suffix_i,  
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts




class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)
        
        return logits

if __name__ == '__main__':

    clip_model = load_clip_to_cpu()
    text_encoder = TextEncoder(clip_model)







