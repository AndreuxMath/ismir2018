import numpy as np
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from scattering_autoencoder.scattering_recurrent import RecurrentScatteringNP
from scattering_autoencoder.utils import create_overlap_add_without_boundaries
from scattering_autoencoder.utils import create_harmonic_overlap_add_without_boundaries

class ScatData(Dataset):
    def __init__(self, J=5, Q=1, N=3, seq_len=int(1e3), dx=2**5,  size_block=64,
                 omega_min=2e-2, omega_max=0.5 * np.pi, num_comp=2, p=0.1,
                 type_process='overlap_add', quantize_target=False,
                 quantization_channels=256, size_epoch=int(1e4),
                 num_harmonics=3,
                 include_haar=False, joint=False, **kwargs):
        # parameters of the dataset class
        self.length_dataset = size_epoch
        # parameters of the scattering
        self.J = J
        self.scat = RecurrentScatteringNP(J, Q, N, only_J=True,
                                          include_haar=include_haar,
                                          joint=joint)
        self.seq_len = seq_len
        self.dx = dx
        # parameters for the process
        self.type_process = type_process
        if not(type_process in ['bernoulli', 'overlap_add',
                                'harmonic_overlap_add']):
            raise ValueError('Unknown process:', type_process)
        self.size_block = size_block
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.num_comp = num_comp
        self.num_block = int(np.ceil(2 * seq_len / size_block) + 1)
        self.num_harmonics = num_harmonics
        self.p = 0.1
        # parameters of the quantization
        self.quantize_target = False
        self.quantization_channels = quantization_channels

    def __len__(self): 
        """
        The dataset is virtually infinite, but this function is implemented so
        as to be compatible with the Dataset class API
        """
        return self.length_dataset

    def _create_example(self):
        # generate a random example
        if self.type_process == 'overlap_add':
            x = create_overlap_add_without_boundaries(
                self.size_block, self.num_comp, 1, normalize=True,
                num_iter=self.num_block, log_sample=True,
                omega_min=self.omega_min, omega_max=self.omega_max)
        elif self.type_process == 'bernoulli':
            x = np.asarray(
                np.random.rand(self.seq_len) < self.p,
                dtype=float)
        elif self.type_process == 'harmonic_overlap_add':
            x = create_harmonic_overlap_add_without_boundaries(
                self.size_block, self.num_harmonics, 1,
                num_iter=self.num_block, log_sample=True,
                omega_min=self.omega_min, omega_max=self.omega_max)
        assert x.shape[0] >= self.seq_len  # to avoid errors
        x = np.squeeze(x[:self.seq_len])  # truncate it to seq_len!
        return x

    def _compute_scat_subsampled(self, x):
        t_send = 3 * (2**self.J)
        # Precompute the scattering at the start, to avoid boundary issues
        hidden = None
        S, hidden = self.scat.forward(x[:t_send], hidden_past=hidden,
                                      return_last_hidden=True)
        S_sub_acc = []
        times_acc = []
        for t in range(t_send, self.seq_len, self.dx):
            # send a block of size dx
            S, hidden = self.scat.forward(x[t:t + self.dx], hidden_past=hidden,
                                          return_last_hidden=True)
            # take the first value, and the corresponding time index
            S_sub_acc.append(np.take(S[self.J], np.array([0]), axis=0))
            times_acc.append(t)
        S_sub = np.concatenate(S_sub_acc, axis=0)
        S_sub = S_sub.transpose()  # channels x time
        return S_sub, times_acc

    def __getitem__(self, idx):
        # create an example
        x = self._create_example()
        # compute its scattering
        S, times_acc = self._compute_scat_subsampled(x)
        times_S = np.arange(0, times_acc[-1] - times_acc[0] + 1,
                            self.dx, dtype=int)
        target = x[times_acc[0]:times_acc[-1] + 1].reshape(1, -1)  # C x T
        # move them to pytorch with the correct types
        S = torch.from_numpy(S).float()
        target = torch.from_numpy(target).float()
        times_S = torch.from_numpy(times_S).long()
        # return them
        return S, target, times_S


class ScatDataFromFile(Dataset):
    def __init__(self, prefix_files=''):
        self.S = torch.load(prefix_files + '_scatterings.pth')
        self.x = torch.load(prefix_files + '_signals.pth')
        self.t = torch.load(prefix_files + '_times.pth')

    def __len__(self):
        return self.S.size(0)

    def __getitem__(self, idx):
        return self.S[idx], self.x[idx], self.t[idx]


def create_dataloader_scattering(params):
    if 'prefix_files' in params:
        sd = ScatDataFromFile(prefix_files=params['prefix_files'])
    else:
        sd = ScatData(**params)
    dataloader = DataLoader(
        sd, batch_size=params['batch_size'], num_workers=params['num_workers'],
        drop_last=True, shuffle=False)
    return dataloader


def create_generator_examples_gan(J=5, Q=1, N=3, seq_len=int(1e3),
                                  batch_size=2, dx=2**5, size_block=64,
                                  is_cuda=False, omega_min=2e-2,
                                  omega_max=0.5 * np.pi, timing=False,
                                  num_comp=2, p=0.1,
                                  type_process="overlap_add",
                                  quantize_target=False,
                                  quantization_channels=256):
    """
    DEPRECATED: see the version with DataLoader!
    """
    # Create the scattering
    if timing:
        tic = time.time()
    scat = RecurrentScattering(J, Q, N, only_J=True)
    if timing:
        print('Scat initialized in', time.time() - tic, 's')
    if is_cuda:
        scat.cuda()
    # Create the parameters
    t_send = 2**(J + 1)
    num_block = int(np.ceil(2 * seq_len / size_block) + 1)
    # main loop, which yields examples
    while True:
        # Create a random signal of adequate size
        if timing:
            tic = time.time()
        if type_process == 'overlap_add':
            x = create_overlap_add_without_boundaries(
                size_block, num_comp, batch_size, normalize=True,
                num_iter=num_block, log_sample=True, omega_min=omega_min,
                omega_max=omega_max)
            x = x[:seq_len]  # truncate it to seq_len!
        elif type_process == 'bernoulli':
            x = np.asarray(np.random.rand(seq_len, batch_size) < p,
                           dtype=float)
        else:
            raise ValueError('Unknown process:' + str(type_process))
        assert x.shape[0] >= seq_len  # to avoid errors
        if timing:
            print('New example created in', time.time() - tic)

        # make it as an torch tensor, move it to cuda if necessary
        if timing:
            tic = time.time()
        x = torch.from_numpy(x)
        if is_cuda:
            x = x.cuda()
        if timing:
            print('TH + cuda in', time.time() - tic)

        # Precompute the scattering at the start, to avoid boundary issues
        hidden = None
        S, hidden = scat.forward(x[:t_send], hidden_past=hidden,
                                 return_last_hidden=True)
        S_sub_acc = []
        times_acc = []
        for t in range(t_send, seq_len, dx):
            # send a block of size dx
            S, hidden = scat.forward(x[t:t + dx], hidden_past=hidden,
                                     return_last_hidden=True)
            # take the first value, and the corresponding time index
            S_sub_acc.append(S[J].narrow(0, 0, 1))
            times_acc.append(t)

        S_subsampled = torch.cat(S_sub_acc, dim=0).permute(1, 2, 0)  # BxCxT
        target = x[times_acc[0]:times_acc[-1]].unsqueeze(-1)  # TxBxC
        times_S = np.arange(0, times_acc[-1] - times_acc[0] + 1, dx, dtype=int)
        yield Variable(S_subsampled.float()), Variable(target.float()), times_S
