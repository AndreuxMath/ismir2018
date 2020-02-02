import torch
import numpy as np

"""
TODO: IMPROVE BOTH FUNCTIONS BY REMOVING ALL NP CALLS!!! AND USING A NEW TO
AVOID THE CUDA QUERY
"""


def create_sig_with_lags(input_seq, k_range, J, ind_end, T, is_cuda=False):
    if type(k_range) != dict:
        myk_range = {j: k_range for j in range(1, J + 1)}
    else:
        myk_range = k_range
    input_seq_j = {}
    for j in range(1, J + 1):
        tocat = []
        # print('\t', j)
        for k in myk_range[j]:
            # print('\t\t', k)
            # inds = input_seq.new((T,)).long()
            # inds = torch.arange(T - 1, -1, -1, out=inds)
            # inds = ind_end - k * (2 ** j) - inds
            # Old numpy version:
            inds = ind_end - int(k * (2**j)) - np.arange(T, dtype=int)[::-1]
            inds = torch.LongTensor(inds)
            if is_cuda:
                inds = inds.cuda()
            # print('\t\t\t', inds)
            newshape = (T,) + input_seq.size()[1:] + (1,)
            tocat.append(input_seq.index_select(0, inds).view(newshape))
            # print('\t\t\t', tocat[-1].shape)
        input_seq_j[float(j)] = torch.cat(tocat, dim=-1)
    return input_seq_j


def create_x_past(input_seq, T, ind_end, n, is_cuda=False):
    tocat = []
    for k in range(n):
        # inds = input_seq.new((T,)).long()
        # inds = torch.arange(T - 1, -1, -1, out=inds)
        # inds = ind_end - k - inds
        inds = ind_end - k - np.arange(T)[::-1]
        inds = torch.LongTensor(inds)
        if is_cuda:
            inds = inds.cuda()
        temp = input_seq.index_select(0, inds)
        temp = temp.view((T,) + input_seq.size()[1:] + (1,))
        tocat.append(temp)
    input_seq_past = torch.cat(tocat, dim=-1)
    return input_seq_past
