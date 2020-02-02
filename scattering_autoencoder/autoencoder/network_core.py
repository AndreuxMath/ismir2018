import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.batchnorm import BatchNorm1d


class Block(nn.Module):
    """
    This block performs 3 operations:
    1) Upsampling, by adding zeros in the time dimension
    2) Temporal convolution
    3) Batch normalization
    4) ReLU element-wise
    """
    def __init__(self, channels_in=1, channels_out=1, kernel_size=1,
                 use_batch_norm=True, apply_relu=True):
        super(Block, self).__init__()
        self.conv = nn.Conv1d(channels_in, channels_out, kernel_size)
        self.relu = nn.ReLU()
        self.batchnorm = BatchNorm1d(channels_out)
        self.use_batch_norm = use_batch_norm
        self.apply_relu = apply_relu

    def forward(self, x):
        # x must have size batch x channels_in x time
        # 1) upsample x by filling it with zeros
        target_size = x.size()[:-1] + (2 * x.size(-1),)
        x_up = x.unsqueeze(-1)  # add one last dim
        x_up = torch.cat(
            [x_up, x_up.new(x_up.size()).fill_(0.)],
            dim=-1).view(target_size)
        # x_up has size batch x channels x 2*time
        # 2) perform a convolution
        y = self.conv(x_up)
        # y has size batch x channels x (2 * time - kernel_size + 1)
        if self.use_batch_norm:
            # 3) apply the batch norm if necessary
            y = self.batchnorm(y)
        # 4) apply the relu, if required
        if hasattr(self, "apply_relu"):
            if self.apply_relu:
                y = self.relu(y)
        else:  # retrocompatibility case
            y = self.relu(y)
        return y


class ConvGen(nn.Module):
    """
    Cascade of blocks:
    Block j takes S_j and maps it to S_{j-1}, after upsampling,
    convolution and non-linearity
    """
    def __init__(self, J, channels_sizes, kernel_size, use_batch_norm=None,
                 last_convolution=False, last_sigmoid=False,
                 bias_last_convolution=True, relu_last_block=True):
        super(ConvGen, self).__init__()
        if use_batch_norm is None:
            do_batch_norm = {j: True for j in range(J, 0, -1)}
        else:
            do_batch_norm = use_batch_norm
        if relu_last_block:
            do_relu_block = {j: True for j in range(J, 0, -1)}
        else:
            do_relu_block = {j: True for j in range(J, 1, -1)}
            do_relu_block[1] = False
            if use_batch_norm is not None:
                if use_batch_norm[1]:
                    print('Warning: removing batch norm for block 1')
            do_batch_norm[1] = False
        self.J = J
        self.blocks = {}
        for j in range(J, 0, -1):
            self.blocks[j] = Block(channels_in=channels_sizes[j],
                                   channels_out=channels_sizes[j - 1],
                                   kernel_size=kernel_size,
                                   use_batch_norm=do_batch_norm[j],
                                   apply_relu=do_relu_block[j])
            acc = 0
            for p in self.blocks[j].parameters():
                self.register_parameter('blocks_' + str(j) + '_' + str(acc), p)
                acc += 1
        self.do_last_convolution = last_convolution
        if last_convolution:
            self.conv1x1 = nn.Conv1d(channels_sizes[0], channels_sizes[-1],
                                     1, padding=0, bias=bias_last_convolution)
        self.do_last_sigmoid = last_sigmoid
        if last_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq, to_pad_left=None):
        current_output = input_seq
        # Look through the layers
        for j in range(self.J, 0, -1):
            # Concatenation with the previous output, if provided
            if to_pad_left is not None:
                current_output = torch.cat(
                    [to_pad_left[j], current_output], dim=-1)
            # go through the block
            current_output = self.blocks[j].forward(current_output)
        if self.do_last_convolution:
            current_output = self.conv1x1(current_output)
        if self.do_last_sigmoid:
            current_output = self.sigmoid(current_output)
        return current_output

    def cuda(self):
        # move manually each block
        for k in self.blocks.keys():
            self.blocks[k].cuda()
        return super(ConvGen, self).cuda()

    def cpu(self):
        # move manually each block
        for k in self.blocks.keys():
            self.blocks[k].cpu()
        return super(ConvGen, self).cpu()


def compute_time_block(time_j_up, dx_j, past_size, look_ahead):
    # intertwin zeros within time_j_up
    newtime = np.concatenate([time_j_up.reshape(-1, 1),
                              (time_j_up + dx_j).reshape(-1, 1)],
                             axis=-1)
    newtime = newtime.reshape(-1)
    return newtime[past_size:-look_ahead + 1]


def compute_time_convgen(time_J, J, past_size, look_ahead):
    dx = 2**J
    times = {J: time_J}
    for j in range(J - 1, -1, -1):
        dx = int(dx / 2)
        times[j] = compute_time_block(times[j + 1], dx, past_size, look_ahead)
    return times


class ScatPredictor(nn.Module):
    def __init__(self, channel_size, kernel_size):
        super(ScatPredictor, self).__init__()
        self.predictor = nn.Conv1d(channel_size, channel_size,
                                   kernel_size, padding=0)

    def forward(self, s_past):
        s_future = self.predictor(s_past)
        return s_future
