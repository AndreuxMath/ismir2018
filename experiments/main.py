import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.cuda as cuda
import json
import numpy as np
import pdb
from argparse import ArgumentParser

from scattering_autoencoder.utils import get_timestamp, get_git_revision_hash
from scattering_autoencoder.autoencoder import ConvGen, train_model
from scattering_autoencoder.scattering_recurrent import compute_size_scattering
from scattering_autoencoder.datasets import create_dataloader_real


CUDA_IS_AVAILABLE = cuda.is_available()
print("Cuda available: {}".format(CUDA_IS_AVAILABLE))


def basic_saving(params):
    prefix_save = params['prefix_save']
    gitref = get_git_revision_hash()
    with open(prefix_save + '_gitref.txt', 'w') as outfile:
        outfile.write(gitref)
        outfile.write('\nmain.py')
    with open(prefix_save + '_params.json', 'w') as outfile3:
        json.dump(params, outfile3)  # json can be read more easily!


if __name__ == '__main__':
    timestamp = get_timestamp()

    parser = ArgumentParser()
    parser.add_argument("--argfile", type=str)

    subargs = parser.parse_args()
    with open(subargs.argfile, "r") as f:
        params = json.load(f)

    params["prefix_save"] = params["path_save"] + "/"  + timestamp
    params["prefix_files"] = params["prefix_data"] + params["timestamp_data"]
    params["timestamp"] = timestamp  # just to record it

    if not os.path.exists(params["path_save"]):
        os.makedirs(params["path_save"])

    # Check the parameters
    assert params['dx'] == 2 ** params['J']
    assert params['Q_loss'] == params['Q']
    if params['maximal_size_dataset'] is not None:
        assert params['maximal_size_dataset'] % params['batch_size'] == 0

    # Computation of the sizes of the channels
    channel_J = compute_size_scattering(
        params['J'], params['Q'], include_haar=params['include_haar'],
        joint=params['joint'], len_fft_joint=params['len_fft_joint'])
    channel_1 = params["size_channel_1"]  # to provide a similar value, but without one layer
    channels_sizes = {params['J']: channel_J, 1: channel_1}
    q = np.power(float(channel_J) / float(channel_1),
                 1. / float(params['J'] - 1))
    for j in range(params['J'] - 1, 1, -1):
        channels_sizes[j] = int(np.ceil(channel_1 * np.power(q, j - 1)))
    channels_sizes[0] = 1
    params['channels_sizes'] = channels_sizes

    # Creation of the generator
    gen = create_dataloader_real(params)
    print('Generator created')

    # Initialization of the networks
    kernel_size = params['look_ahead'] + params['past_size']
    cg = ConvGen(params['J'], params['channels_sizes'], kernel_size,
                 last_convolution=params['last_convolution'],
                 bias_last_convolution=params['bias_last_convolution'],
                 relu_last_block=params['relu_last_block'])

    # Creation of the optimizer and the criterion
    optimizer = optim.Adam(cg.parameters(), lr=params['lr'],
                           weight_decay=params['weight_decay'])

    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=params['milestones'], gamma=params['gamma'])
    
    # make cg in 'train' mode:
    cg.train()
    
    # Check cuda, if required
    if params['is_cuda']:
        cg = cg.cuda()
    print('Network created')
    print('Timestamp =', params['timestamp'])
    
    # Train the network
    print('Training bottom network')
    try:
        cg, acc_loss = train_model(cg, gen, optimizer, params,
                                   scheduler=scheduler)
    except KeyboardInterrupt as e:
        print('Saving parameters...')
        basic_saving(params)
        raise e
    cg = cg.cpu()
    
    # Save the results
    prefix_save = params['prefix_save']
    torch.save(cg, prefix_save + '_convgen.pth')
    for k in acc_loss.keys():
        np.save(prefix_save + '_loss_bottom_' + k + '.npy', acc_loss[k])

    basic_saving(params)
    print('Saved with timestamp: ', timestamp)
