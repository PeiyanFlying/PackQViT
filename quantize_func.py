"""
    Quantized modules: the base class
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math
from enum import Enum

__all__ = ['Quantizemode',  'Quantize_Conv2d', 'Quantize_Linear', 'Quantize_Activation',
           'trunc', 'get_mask', 'StopGrad']


class Quantizemode(Enum):
    kernel_wise = 2
    layer_wise = 1




def get_mask(param, sparsity):
    bottomk, _ = torch.topk(param.abs().view(-1), int(sparsity * param.numel()), largest=False, sorted=True)
    threshold = bottomk.data[-1]  # This is the largest element from the group of elements that we prune away
    return torch.gt(torch.abs(param), threshold).type(param.type())

def clamp(input, min, max):
    return torch.clamp(input, min, max)



def log_shift(value_fp):
    value_shift = 2 ** (torch.log2(value_fp).ceil())
    return value_shift
def get_quantized_range(num_bits):
    n = 2 ** (num_bits - 1)
    return 0, 2 ** num_bits - 1




def linear_quantize(input, scale_factor):
    return torch.round(scale_factor * input)

def linear_dequantize(input, scale_factor):
    return input / scale_factor


def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max):
    output = linear_quantize(input, scale_factor)
    return clamp(output, clamp_min, clamp_max)





def trunc(fp_data, nbits=8):
    il = torch.log2(torch.max(fp_data.max(), fp_data.min().abs())) + 1
    il = math.ceil(il - 1e-5)
    qcode = nbits - il
    scale_factor = 2 ** qcode
    clamp_min, clamp_max = get_quantized_range(nbits, signed=True)
    q_data = linear_quantize_clamp(fp_data, scale_factor, clamp_min, clamp_max)
    q_data = linear_dequantize(q_data, scale_factor)
    return q_data, qcode


def get_default_kwargs_q(kwargs_q, layer_type):
    default = {
        'nbits': 4
    }
    if isinstance(layer_type, Quantize_Conv2d):
        default.update({
            'mode': Quantizemode.layer_wise})
    elif isinstance(layer_type, Quantize_Linear):
        pass
    elif isinstance(layer_type, Quantize_Activation):
        pass
        # default.update({
        #     'signed': 'Auto'})
    else:
        assert NotImplementedError
        return
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q

class StopGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, stopGradientMask):
        ctx.save_for_backward(stopGradientMask)
        return weight

    @staticmethod
    def backward(ctx, grad_outputs):
        Mask, = ctx.saved_tensors
        grad_inputs = grad_outputs * Mask
        return grad_inputs, None


class Quantize_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
        super(Quantize_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.q_mode = kwargs_q['mode']
        if self.q_mode == Quantizemode.kernel_wise:
            self.alpha = Parameter(torch.Tensor(out_channels))
        else:  # layer-wise quantization
            self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    def extra_repr(self):
        s_prefix = super(Quantize_Conv2d, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)


class Quantize_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs_q):
        super(Quantize_Linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.q_mode = kwargs_q['mode']
        self.alpha = Parameter(torch.Tensor(1))
        if self.q_mode == Quantizemode.kernel_wise:
            self.alpha = Parameter(torch.Tensor(out_features))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(Quantize_Linear, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)


class Quantize_Activation(nn.Module):
    def __init__(self, in_features, **kwargs_q):
        super(Quantize_Activation, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            self.register_parameter('zero_point', None)
            return
        # self.signed = kwargs_q['signed']
        self.q_mode = kwargs_q['mode']
        self.alpha = Parameter(torch.Tensor(1))
        self.zero_point = Parameter(torch.Tensor([0]))
        if self.q_mode == Quantizemode.kernel_wise:
            self.alpha = Parameter(torch.Tensor(in_features))
            self.zero_point = Parameter(torch.Tensor(in_features))
            torch.nn.init.zeros_(self.zero_point)
        # self.zero_point = Parameter(torch.Tensor([0]))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    def extra_repr(self):
        # s_prefix = super(Quantize_Activation, self).extra_repr()
        if self.alpha is None:
            return 'fake'
        return '{}'.format(self.kwargs_q)
