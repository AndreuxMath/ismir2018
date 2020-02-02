import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')  # otherwise, weird bug
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import pdb
from scattering_autoencoder.scattering_recurrent import RecurrentScatteringNP


class ScatDataMusic(Dataset):
    """
    This class inherits from the pytorch Dataset class. It is meant
    to prepare the musical dataset "beethoven" for training purposes,
    in conjunction with a Dataloader for parallelization.

    data = ScatDataMusic(...) (see below for a description of the arguments)
    data[i] returns the scattering, original signal and time indices for
    example i in dataset (train or test)
    """
    def __init__(self, J=5, Q=1, N=3, dx=2**5, include_haar=False,
                 joint=False, len_fft_joint=6, max_length=int(1e5),
                 avg_U1_before_joint=True,
                 path_dataset='../../data/music/music_train_norm2_4096.npy',
                 normalize_amplitude=True, t_send_factor=1, **kwargs):
        """
        J: logarithmic factor for the scattering scale 2^J
        Q: number of wavelets per octave (should be ~12 for music)
        N: polynomial factor for the gammatone wavelets, it controls their
        smoothness (they are of class C^N typically)
        dx: subsampling factor for the scatterings. Typically dx=2^J,
            but another value can be chosen.
        include_haar: whether or not to add a Haar wavelet among all wavelets
            (not analytical, but allows to cover high frequencies)
        joint: perform a joint time-frequency scattering transform
        len_fft_joint: support of the filters along frequencies. Typically,
            len_fft_joint should be chosen of the order of Q.
        max_length: temporal support size which is used to renormalize the
            gammatone wavelets. If it is too small, the code might return
            an error
        avg_U1_before_joint: whether or not to use joint time-frequency filters
            on first order coefficients when joint. If this is false,
            then a frequential filter is used. If True, a product of the
            frequential filters and the temporal averaging phi_J is used.
        path_dataset: path to the dataset (a stored numpy array of dims
            dataset_size x time)
        normalize_amplitude: normalization of the signals in [-1, 1] (by
            dividing by the maximum value)
        t_send_factor: cuts the beginning of the signal at t_send_factor*dx
            because of the zero-padding.
        """
        self.path_dataset = path_dataset
        self.all_waveforms = np.load(self.path_dataset)
        self.normalize_amplitude = normalize_amplitude
        if normalize_amplitude:  # do it ONCE
            self.all_waveforms = self.all_waveforms /\
                np.max(np.abs(self.all_waveforms), axis=1).reshape(-1, 1)

        # parameters of the scattering
        self.J = J
        self.scat = RecurrentScatteringNP(
            J, Q, N, only_J=True, include_haar=include_haar,
            joint=joint, len_fft_joint=len_fft_joint, max_length=max_length,
            avg_U1_before_joint=avg_U1_before_joint)
        self.t_send_factor = t_send_factor
        self.dx = dx

    def __len__(self):
        """
        The dataset is completely finite!
        The size of the dataset doubles if some noise is added,
        because we may - or may not - add noise to this sample
        """
        return self.all_waveforms.shape[0]

    def _load_example(self, idx):
        """
        Returns example idx of the dataset (stored in all_waveforms)
        """
        return self.all_waveforms[idx]

    def _compute_scat_subsampled(self, x):
        """
        Computes a recurrent gammatone scattering for example x,
        subsampled by a factor dx
        Returns:
        S_sub: the subsampled scatterings [scat_channels x time]
        times_acc: numpy array of size [time], containing the absolute
            times at which S_sub was computed
        """
        t_send = self.t_send_factor * (2**self.J)
        # Precompute the scattering at the start, to avoid boundary issues
        hidden = None
        S, hidden = self.scat.forward(x[:t_send], hidden_past=hidden,
                                      return_last_hidden=True)
        S_sub_acc = []
        times_acc = []
        for t in range(t_send, x.size, self.dx):
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
        """
        Returns the scattering, signal and times for example idx
        S: subsampled scattering (torch FloatTensor channels x time)
        target: original signal torch FloatTensor 1 x time
        times_S: times of the scattering, relative to the original cut.
            times_S starts at 0 and the corresponding time indices for
            target are also assumed to start at 0.
        """
        # create an example
        x = self._load_example(idx)
        # depending on the parity of idx, add or do not add noise
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


class ScatDataMusicFromFile(Dataset):
    """
    Dataset which exploits a precomputed dataset, using ScatDataMusic
    """
    def __init__(self, prefix_files=''):
        self.S = torch.load(prefix_files + '_scatterings.pth')
        self.x = torch.load(prefix_files + '_signals.pth')
        self.t = torch.load(prefix_files + '_times.pth')

    def __len__(self):
        return self.S.size(0)

    def __getitem__(self, idx):
        return self.S[idx], self.x[idx], self.t[idx]


def create_dataloader_music(params):
    """
    Creates a dataloader based on a dataset which is either ScatDataMusic
    or ScatDataMusicFromFile.
    Params is a dictionary which should contain the following keys:
        -if using files, simply 'prefix_files' as expected by
         ScatDataMusicFromFile
        -if recomputing all scatterings, then all the arguments expected
         by ScatDataMusic, with their names
    In all cases, params should contain the standard parameters for
    DataLoaders:
    - batch_size: size of each batch to return (in the case of ScatDataMusic,
        this value should be kept to 1)
    - num_workers: number of workers used to retrieve samples. Can be kept
        to 1 in case of a file. Warning, if num_workers > 1 and ScatDataMusic
        exploits a random number generation, the seed might be the same for all
        workers.
    """
    if 'prefix_files' in params:
        sd = ScatDataMusicFromFile(prefix_files=params['prefix_files'])
    else:
        sd = ScatDataMusic(**params)
    dataloader = DataLoader(
        sd, batch_size=params['batch_size'], num_workers=params['num_workers'],
        drop_last=True, shuffle=False)
    return dataloader


def create_dataset_music(params):
    """
    Creates a musical dataset from the raw signals.
    It proceeds in 2 steps:
    1/ Iteration over the dataset (possibly in parallel thanks to DataLoader)
        to compute all scatterings, using ScatDataMusic
    2/ Cutting the resulting signals at the adequate size, and keeping the meta
        information concerning this creation.
    Then, the resulting dataset is saved.

    params is a dictionary containing the keys for all subsequent calls:
        - create_dataloader_music
        - step 2/, which involves params['seq_len_S'], which is the maximal
        temporal extent which is accepted for scattering vectors.
        The signals are kept accordingly.
        There might be an overlap between some samples at the last value,
        in order to prevent losing some parts of the signal.
        - 'path_save': folder (with / at the end)
        - 'timestamp': identifier for the dataset (name)
    """
    # create the dataloader
    dataloader = create_dataloader_music(params)
    # Precomputing all the scatterings in parallel:
    # we simply hack the dataloader multiprocessing capabilities
    S_acc, x_acc, t_acc = [], [], []
    n_iter = 0
    for samples in tqdm(dataloader, desc="Scattering computing"):
        S, x, t = samples
        S_acc.append(S)
        x_acc.append(x)
        t_acc.append(t)
        n_iter += 1
    # cut at the right size
    print('Postprocessing...')
    S_all = []
    x_all = []
    t_all = []
    seq_len_S = params['seq_len_S']
    for i in range(len(S_acc)):
        k = S_acc[i].size(-1) // seq_len_S
        if k >= 1:  # otherwise, we do not consider this sample!
            for l in range(k):
                S_temp = S_acc[i][...,
                                  l * seq_len_S:(l + 1) * seq_len_S].clone()
                t_temp = t_acc[i][0, l * seq_len_S:(l + 1) * seq_len_S].clone()
                t_start = t_temp[0]
                x_temp = x_acc[i][..., t_temp[0]: t_temp[-1] + 1].clone()
                t_temp -= t_start
                S_all.append(S_temp)
                t_all.append(t_temp.view(1, -1))
                x_all.append(x_temp)
            if S_acc[i].size(-1) % seq_len_S != 0:
                # we take the last one
                S_temp = S_acc[i][..., -seq_len_S:].clone()
                t_temp = t_acc[i][0, -seq_len_S:].clone()
                t_start = t_temp[0]
                x_temp = x_acc[i][..., t_temp[0]: t_temp[-1] + 1].clone()
                t_temp -= t_start
                S_all.append(S_temp)
                t_all.append(t_temp.view(1, -1))
                x_all.append(x_temp)
    # Concatenate
    S = torch.cat(S_all, dim=0)
    x = torch.cat(x_all, dim=0)
    t = torch.cat(t_all, dim=0)
    # Display
    print('Final number of examples =', S.size(0))
    # make a final permutation to make sure that the examples will not always
    # come from the same example
    perm = np.random.permutation(S.size(0))
    perm_th = torch.from_numpy(perm)
    S = S[perm_th]
    x = x[perm_th]
    t = t[perm_th]
    # save
    print('Saving...')
    path_save = params['path_save'] + params['timestamp']
    torch.save(S, path_save + '_scatterings.pth')
    torch.save(x, path_save + '_signals.pth')
    torch.save(t, path_save + '_times.pth')
    with open(path_save + '_params.json', 'w') as f:
        json.dump(params, f)
    return params['timestamp']
