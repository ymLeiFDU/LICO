
from copy import deepcopy

import torch
import torch.distributed as dist



# class ModelEMA(object):
#     def __init__(self, model, decay):
#         self.ema = deepcopy(model)
#         self.ema = self.ema.cuda()
#         self.ema.eval()
#         self.decay = decay
#         self.ema_has_module = hasattr(self.ema, 'module')
#         # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
#         self.param_keys = [k for k, _ in self.ema.named_parameters()]
#         self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
#         for p in self.ema.parameters():
#             p.requires_grad_(False)

#     def update(self, model):
#         needs_module = hasattr(model, 'module') and not self.ema_has_module
#         with torch.no_grad():
#             msd = model.state_dict()
#             esd = self.ema.state_dict()
#             for k in self.param_keys:
#                 if needs_module:
#                     j = 'module.' + k
#                 else:
#                     j = k
#                 model_v = msd[j].detach()
#                 ema_v = esd[k]
#                 esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

#             for k in self.buffer_keys:
#                 if needs_module:
#                     j = 'module.' + k
#                 else:
#                     j = k
#                 esd[k].copy_(msd[j])



"""
为什么 param 和 buffer 要采用不同的的更新策略
param 是 指数移动平均数，buffer 不是
"""


class EMA(object):
    def __init__(self, model, alpha=0.999):
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        # num_batches_tracked, running_mean, running_var in bn
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        # decay = min(self.alpha, (self.step + 1) / (self.step + 10))  # ????
        decay = self.alpha
        state = self.model.state_dict()  # current params
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name] + (1 - decay) * state[name]
            )
        # for name in self.buffer_keys:
        #     self.shadow[name].copy_(
        #         decay * self.shadow[name]
        #         + (1 - decay) * state[name]
        #     )

        self.step += 1

    def update_buffer(self):
        # without EMA
        state = self.model.state_dict()
        for name in self.buffer_keys:
            self.shadow[name].copy_(state[name])

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }