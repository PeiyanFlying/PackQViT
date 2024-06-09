import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import Mlp
from timm.models.layers.drop import DropPath
from timm.models.layers.helpers import to_2tuple
from timm.models.layers.weight_init import trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

import numpy as np
from Quant import *
from _quan_funct import *
from timm.models import create_model
from quant_vit import lowbit_VisionTransformer, _cfg

def fourbits_deit_tiny_patch16_224(pretrained=True, **kwargs):
    model = lowbit_VisionTransformer(
        nbits=4, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = checkpoint = torch.load('deit_t_best_checkpoint_4bit.pth',map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

model = fourbits_deit_tiny_patch16_224()
# checkpoint = torch.load('deit_t_best_checkpoint_4bit.pth',map_location="cpu")
# print(checkpoint["model"])
print(model)

