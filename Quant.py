import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.autograd import Function, Variable
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from quantize_func import Quantize_Conv2d, Quantizemode, Quantize_Linear, Quantize_Activation


__all__ = ['Conv2dQuantize', 'LinearQuantize', 'ActQuantize', 'QIntSoftmax', 'LayerNormActQ']


class FunQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        indicate_middle = 1.0 - indicate_small - indicate_big  # Thanks to @haolibai
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
            -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        # The following operation can make sure that alpha is always greater than zero in any case and can also
        # suppress the update speed of alpha. (Personal understanding)
        # grad_alpha.clamp_(-alpha.item(), alpha.item())  # FYI
        return grad_weight, grad_alpha, None, None, None


def grad_scaling(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_sect(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class Conv2dQuantize(Quantize_Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=8, mode=Quantizemode.kernel_wise, **kwargs):
        super(Conv2dQuantize, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, mode=mode)
        self.act = ActQuantize(in_features=in_channels, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() * 2)
            self.init_state.fill_(1)
        """  
        Implementation according to paper. 
        Feels wrong ...
        When we initialize the alpha as a big number (e.g., self.weight.abs().max() * 2), 
        the clamp function can be skipped.
        Then we get w_q = w / alpha * alpha = w, and $\frac{\partial w_q}{\partial \alpha} = 0$
        As a result, I don't think the pseudo-code in the paper echoes the formula.
       
        Please see jupyter/STE_LSQ.ipynb fo detailed comparison.
        """
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scaling(self.alpha, g)
        # print(alpha.shape)
        # print(self.weight.shape)
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w_q = round_sect((self.weight / alpha).clamp(Qn, Qp)) * alpha

        x = self.act(x)
        # w = w.clamp(Qn, Qp)
        # q_w = round_sect(w)
        # w_q = q_w * alpha

        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LinearQuantize(Quantize_Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
        super(LinearQuantize, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, mode=Quantizemode.kernel_wise)
        self.act = ActQuantize(in_features=in_features, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1:
        alpha = grad_scaling(self.alpha, g)
        alpha = alpha.unsqueeze(1)
        w_q = round_sect((self.weight / alpha).clamp(Qn, Qp)) * alpha

        x = self.act(x)
        # w = self.weight / alpha
        # w = w.clamp(Qn, Qp)
        # q_w = round_sect(w)
        # w_q = q_w * alpha

        # Method2:
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.linear(x, w_q, self.bias)


class ActQuantize(Quantize_Activation):
    def __init__(self, in_features, nbits_a=4, mode=Quantizemode.kernel_wise, **kwargs):
        super(ActQuantize, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
        self.alpha_q = self.alpha
        # print(self.alpha.shape, self.zero_point.shape)
    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            # The init alpha for activation is very very important as the experimental results shows.
            # Please select a init_rate for activation.
            # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                Qn = 0
                Qp = 2 ** self.nbits - 1
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.zero_point.data.copy_(self.zero_point.data * 0.9 + 0.1 * (torch.min(x.detach()) - self.alpha.data * Qn))
            self.init_state.fill_(1)
        # print(self.alpha.size()) channel_dim

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        alpha = grad_scaling(self.alpha, g)
        zero_point = grad_scaling(zero_point, g)
        # x = round_sect((x / alpha).clamp(Qn, Qp)) * alpha
        if len(x.shape)==2:
            alpha = alpha.unsqueeze(0)
            zero_point = zero_point.unsqueeze(0)
        elif len(x.shape)==4:
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = round_sect((x / alpha + zero_point).clamp(Qn, Qp))
        x = (x - zero_point) * alpha
        self.alpha_q = nn.Parameter(alpha)

        return x

class QIntSoftmax(nn.Module):

    def __init__(self,
                 # log_i_softmax=False,
                 # quant=False,
                 # calibrate=False,
                 # last_calibrate=False,
                 bit = 8):
        super(QIntSoftmax, self).__init__()

        # self.log_i_softmax = log_i_softmax
        # self.quant = quant
        # self.calibrate = calibrate
        # self.last_calibrate = last_calibrate
        self.bits = bit
        # self.bit_type = bit_type
        # self.calibration_mode = calibration_mode
        # self.observer_str = observer_str
        # self.quantizer_str = quantizer_str

        # self.module_type = 'activation'
        # self.observer = build_observer(self.observer_str, self.module_type,
        #                                self.bit_type, self.calibration_mode)
        # self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
        #                                  self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        x_log_floor = floor_ste.apply(x.log2())
        big = x_log_floor.clone()
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor):

        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = floor_ste.apply(coef[1] / scaling_factor)
            c_int = floor_ste.apply(coef[2] / scaling_factor**2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor**2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.6931  # -ln2
            n = 30  # sufficiently large integer
            x0_int = floor_ste.apply(x0 / scaling_factor)
            x_int = torch.max(x_int, n * x0_int)
            q = floor_ste.apply(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = torch.clamp(floor_ste.apply(exp_int * 2**(n - q)), min=0)
            scaling_factor = exp_scaling_factor / 2**n
            return exp_int, scaling_factor

        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        return exp_int, exp_int_sum

    def forward(self, x, scale):
        if scale is not None:
            exp_int, exp_int_sum = self.int_softmax(x, scale)
            softmax_out = round_ste.apply(exp_int_sum / exp_int)
            rounds = self.log_round(softmax_out)
            mask = rounds >= 2**self.bits
            qlog = torch.clamp(rounds, 0, 2**self.bits - 1)
            deq_softmax = 2**(-qlog)
            deq_softmax[mask] = 0
            return deq_softmax
        else:
            x = x.softmax(dim=-1)
            deq_softmax = Log2_Quantize.apply(x, self.bits, 'po2')
            # deq_softmax[mask] = 0
            # if self.calibrate:
            #     self.quantizer.observer.update(x)
            #     if self.last_calibrate:
            #         self.quantizer.update_quantization_params(x)
            # if not self.quant:
            #     return x
            # x = self.quantizer(x)
            return deq_softmax

class floor_ste(Function):
    """
    Straight-through Estimator(STE) for torch.floor()
    """
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class round_ste(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()



class Log2_Quantize(Function):

    @staticmethod
    def forward(ctx, input, bit, scheme='po2'):
        # I. fix point:
        if scheme == 'fp':
            scale = float(2 ** bit-1)
            out = torch.round(input * scale) / scale

        # II. power of 2:
        elif scheme == 'po2':
            out = 2 ** torch.round(torch.log2(input)) * (input > 2 ** (-2 ** bit + 1)).float()

        # III. sp2:
        elif scheme == 'sp2':
            size = input.size()
            y = input.reshape(-1)

            centroids = torch.tensor(
                [0, 2 ** -4, 2 ** -3, 2 ** -4 + 2 ** -3, 2 ** -2, 2 ** -2 + 2 ** -3, 2 ** -1, 2 ** -1 + 2 ** -3,
                 1]).cuda()
            mag = y - centroids.reshape(-1, 1)

            minimum = torch.min(torch.abs(mag), dim=0)[1]
            out = centroids[minimum]
            out = out.reshape(size)
        else:
            raise NotImplementedError
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class LayerNormActQ(Quantize_Activation):
    def __init__(self, in_features, nbits_a=4, mode=Quantizemode.kernel_wise, **kwargs):
        super(LayerNormActQ, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
        self.alpha_q = self.alpha
        self.eps = torch.finfo(torch.float32).eps
        # print(self.alpha.shape, self.zero_point.shape)
    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            # The init alpha for activation is very very important as the experimental results shows.
            # Please select a init_rate for activation.
            # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                Qn = 0
                Qp = 2 ** self.nbits - 1
            max_val_t = x.max()
            # print(max_val_t[0].size())
            min_val_t = x.min()
            scale8 = (max_val_t - min_val_t) / float(Qp - Qn)
            scale8.clamp_(self.eps)
            scale4 = scale8 / 2
            scale2 = scale4 / 2
            scale1 = scale2 / 2
            zero_point = Qn - torch.round(min_val_t / scale8) # scale8 batch*198*1
            zero_point.clamp_(Qn, Qp)
            scale_mask = torch.ones_like(x[0][0]) #192
            for j in range(x.shape[2]):
                data = x[..., j].unsqueeze(-1)
                data_q1 = ((data / scale1 + zero_point).round().clamp(Qn, Qp) -
                           zero_point) * scale1
                data_q2 = ((data / scale2 + zero_point).round().clamp(Qn, Qp) -
                           zero_point) * scale2
                data_q4 = ((data / scale4 + zero_point).round().clamp(Qn, Qp) -
                           zero_point) * scale4
                data_q8 = ((data / scale8 + zero_point).round().clamp(Qn, Qp) -
                           zero_point) * scale8
                score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
                score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
                score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
                score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
                score = [score1, score2, score4, score8]
                scale_mask[j] *= 2 ** score.index(min(score))
            self.alpha.data.copy_(scale1 * scale_mask) # value * mask(192)
            self.zero_point.data.copy_(zero_point)
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)
        # print(self.alpha.size()) 192

        # Method1:
        zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        alpha = grad_scaling(self.alpha, g)
        zero_point = grad_scaling(zero_point, g)
        # print(alpha.size()) 192
        # x = round_sect((x / alpha).clamp(Qn, Qp)) * alpha
        if len(x.shape)==2:
            alpha = alpha.unsqueeze(0)
            zero_point = zero_point.unsqueeze(0)
        elif len(x.shape)==4:
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        #####
        x = round_sect((x / alpha + zero_point).clamp(Qn, Qp))
        x = (x - zero_point) * alpha
        self.alpha_q = nn.Parameter(alpha)
        # print(alpha.size()) 192

        return x

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()
