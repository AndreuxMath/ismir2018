import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function


def apply_func_at_some_coords(v, func):
    m = v.size(1)
    if m > 1:
        return torch.cat(
            (v.narrow(1, 0, 1), func(v.narrow(1, 1, m - 1))), dim=1)
    else:
        return v


def pad1D(x, pad_left, pad_right, mode='constant', value=0):
    return F.pad(x.unsqueeze(2),
                 (pad_left, pad_right, 0, 0),
                 mode=mode, value=value).squeeze(2)


class WeightedMSELoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedMSELoss, self).__init__()
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        if self.weight is None:
            return F.mse_loss(input, target)
        else:
            return F.mse_loss(input * Variable(self.weight),
                              target * Variable(self.weight))


class ModulusStable(Function):

    @staticmethod
    def forward(ctx, input, p=2, dim=-1, keepdim=False):
        ctx.p = p
        ctx.dim = dim
        ctx.keepdim = False if keepdim is None else keepdim

        if dim is None:
            norm = input.norm(p)
            output = input.new((norm,))
        else:
            if keepdim is not None:
                output = input.norm(p, dim, keepdim=keepdim)
            else:
                output = input.norm(p, dim)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_variables
        if ctx.dim is not None and ctx.keepdim is False and input.dim() != 1:
            grad_output = grad_output.unsqueeze(ctx.dim)
            output = output.unsqueeze(ctx.dim)

        if ctx.p == 2:
            grad_input = input.mul(grad_output).div(output)
        else:
            input_pow = input.abs().pow(ctx.p - 2)
            output_pow = output.pow(ctx.p - 1)
            grad_input = input.mul(input_pow).mul(grad_output).div(output_pow)

        # Special case at 0 where we return a subgradient containing 0
        grad_input.masked_fill_(output == 0, 0)

        return grad_input, None, None, None


def dictionary_to_tensor(h):
    # compute the size of the array
    any_key = list(h.keys())[0]
    out = torch.zeros(
        (len(h.keys()),) + h[any_key].size()).type(type(h[any_key]))
    for i, k in enumerate(sorted(list(h.keys()))):
        out[i] = h[k]
    return out
