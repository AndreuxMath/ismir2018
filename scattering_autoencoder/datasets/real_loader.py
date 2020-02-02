import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')  # otherwise, weird bug
import numpy as np
import time
import torch
import os
from torch.utils.data import Dataset, DataLoader
from librosa.core import load as loadwav
import json
import pickle
from tqdm import tqdm
import pdb
from scattering_autoencoder.scattering_recurrent import RecurrentScatteringNP


def get_files_timit(path, **kwargs):
    """
    Explores the TIMIT folder to retrieve all wav addresses
    """
    all_files = {}
    regions = os.listdir(path)
    for id_region in range(len(regions)):
        speakers = os.listdir(os.path.join(path, regions[id_region]))
        for id_speaker in range(len(speakers)):
            subdir = os.path.join(path, regions[id_region],
                                  speakers[id_speaker])
            files = [f for f in os.listdir(subdir) if '.WAV' in f]
            for f in files:
                prefix = str.split(f, '.')[0]
                key = (regions[id_region], speakers[id_speaker], prefix)
                val = os.path.join(path, regions[id_region],
                                   speakers[id_speaker], f)
                all_files[key] = val
    return all_files


def get_files_vctk(path_dataset, is_train_dataset=True, id_split=None,
                   **kwargs):
    """
    Explores the VCTK folder
    """
    if is_train_dataset:
        suffix = '_train'
    else:
        suffix = '_test'
    suffix = suffix + '_examples'
    if id_split is not None:
        suffix = suffix + '_' + str(id_split)
    print(suffix)
    suffix = suffix + '.pkl'
    with open(path_dataset + suffix, 'rb') as f:
        all_files = pickle.load(f)
    return all_files


def get_files_nsynth(path_dataset, family='keyboard', source='acoustic',
                     **kwargs):
    """
    Here path_dataset is meant as the /Nsynth/nsynth-train/ path
    """
    with open(os.path.join(path_dataset, 'examples.json'), 'r') as f:
        meta = json.load(f)
    all_files = {}
    for k in meta.keys():
        if meta[k]['instrument_family_str'] == family:
            if meta[k]['instrument_source_str'] == source:
                instr = meta[k]['instrument']
                velocity = meta[k]['velocity']
                pitch = meta[k]['pitch']
                key = (instr, velocity, pitch)
                val = os.path.join(path_dataset, 'audio', k + '.wav')
                all_files[key] = val
    return all_files

get_files = {'timit': get_files_timit, 'vctk': get_files_vctk,
             'nsynth': get_files_nsynth}


class ScatDataReal(Dataset):
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
                 path_dataset='../../data/TIMIT/TRAIN/', sr=4096,
                 dataset_name='timit', normalize_amplitude=True,
                 provide_meta=False, t_send_factor=1,
                 add_noise=False, noise_factor=0.02,
                 threshold_silence=None, pad_left=0,
                 seq_len_S=9, truncate_seq=False, **kwargs):
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
        path_dataset: path to the dataset (the base folder, which is then
            used by the corresponding function get_files[dataset_name])
        dataset_name: name of the dataset, to address the adequatae function
        normalize_amplitude: normalization of the signals in [-1, 1] (by
            dividing by the maximum value)
        provide_meta: provide meta information on each file (useful for
            structured datasets such as TIMIT: it contains the name of the
            speaker, region, etc)
        t_send_factor: cuts the beginning of the signal at t_send_factor*dx
            because of the zero-padding.
        add_noise: whether or not to add noise to each signal (the number
            of examples then doubles, as we keep a non-noisy version)
        noise_factor: variance of the noise which is added
        threshold_silence: threshold regions of silence when not None
        pad_left: whether to pad on th left or not
        seq_len_S: maximal length of the sequence
        truncate_seq: whether or not the sequence should be truncated
        """
        self.path_dataset = path_dataset
        self.provide_meta = provide_meta
        self.dataset_name = dataset_name
        if not(dataset_name in get_files.keys()):
            raise ValueError('Unknown dataset ' + str(dataset_name))
        raw_files = get_files[dataset_name](path_dataset, **kwargs)
        # **kwargs allows to transmit arguments, notably
        # the split of the dataset for something else than TIMIT
        all_files = {}
        i = 0
        for k, v in raw_files.items():
            all_files[i] = {'meta': k, 'path': v}
            i += 1
        self.all_files = all_files
        self.sr = sr
        self.normalize_amplitude = normalize_amplitude
        self.add_noise = add_noise
        self.noise_factor = noise_factor
        # NOne: don't do anything, otherwise it should be a number >0
        self.threshold_silence = threshold_silence
        # parameters of the scattering
        self.J = J
        self.scat = RecurrentScatteringNP(
            J, Q, N, only_J=True, include_haar=include_haar,
            joint=joint, len_fft_joint=len_fft_joint, max_length=max_length,
            avg_U1_before_joint=avg_U1_before_joint)
        self.t_send_factor = t_send_factor
        self.dx = dx
        self.pad_left = pad_left
        # truncation on the right: if truncate_seq is True, by construction,
        # we stop the computation of the scattering once we have seq_len_S
        # samples of S (after subsampling)
        self.truncate_seq = truncate_seq
        self.seq_len_S = seq_len_S

    def __len__(self):
        """
        The dataset is completely finite!
        The size of the dataset doubles if some noise is added,
        because we may - or may not - add noise to this sample
        """
        if self.add_noise:
            return 2 * len(self.all_files)
        else:
            return len(self.all_files)

    def _load_example(self, idx):
        """
        Returns example idx of the dataset (stored in all_waveforms)
        Performs the normalization of the amplitudes + trim silence regions
        """
        if self.add_noise:
            actual_idx = idx // 2
        else:
            actual_idx = idx
        # load the corresponding example
        path = self.all_files[actual_idx]['path']
        x, _ = loadwav(path, sr=self.sr)
        x = np.squeeze(x)
        if self.pad_left > 0:
            x = np.concatenate([np.zeros(self.pad_left, dtype=x.dtype), x])
        if self.normalize_amplitude:
            x /= np.max(np.abs(x))
        if self.threshold_silence is not None:
            t_send = self.t_send_factor * (2**self.J)
            ok_inds = np.where(np.abs(x) > self.threshold_silence)
            ind_start = max(np.min(ok_inds) - self.t_send_factor * self.dx,
                            0)
            ind_end = min(np.max(ok_inds) + 1 + self.dx, x.size)
            # cut only in this case
            if ind_end - ind_start > t_send:
                x = x[ind_start:ind_end]
            else:
                # make sure the sample will be rejected by the postfilter
                x = np.zeros(3 * t_send)
        return x

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
        if t_send > 0:  # otherwise, bug!
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
            if self.truncate_seq:
                if len(S_sub_acc) >= self.seq_len_S:
                    break  # stop the loop before the end
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
        if self.add_noise:
            if idx % 2 == 1:
                actual_id = idx // 2
                is_noisy = True
                x_for_S = x + self.noise_factor * np.random.randn(x.size)
            else:
                is_noisy = False
                actual_id = idx // 2
                x_for_S = x
        else:
            is_noisy = False
            actual_id = idx
            x_for_S = x
        # check if x is a true example
        if self.threshold_silence is not None:
            if np.any(np.abs(x) > self.threshold_silence):
                to_keep = True
            else:
                to_keep = False
        else:
            to_keep = True
        # compute its scattering
        S, times_acc = self._compute_scat_subsampled(x_for_S)
        times_S = np.arange(0, times_acc[-1] - times_acc[0] + 1,
                            self.dx, dtype=int)
        target = x[times_acc[0]:times_acc[-1] + 1].reshape(1, -1)  # C x T
        # move them to pytorch with the correct types
        S = torch.from_numpy(S).float()
        target = torch.from_numpy(target).float()
        times_S = torch.from_numpy(times_S).long()
        # return them
        if self.provide_meta:
            new_meta = self.all_files[actual_id]['meta'] +\
                (is_noisy, to_keep,)
            return S, target, times_S, new_meta
        else:
            return S, target, times_S


class ScatDataRealFromFile(Dataset):
    """
    Dataset which exploits a precomputed dataset, using ScatDataReal
    """
    def __init__(self, prefix_files='', meta=False):
        self.S = torch.load(prefix_files + '_scatterings.pth')
        self.x = torch.load(prefix_files + '_signals.pth')
        self.t = torch.load(prefix_files + '_times.pth')
        if meta:
            with open(prefix_files + '_meta.json', 'r') as f:
                self.meta = json.load(f)

    def __len__(self):
        return self.S.size(0)

    def __getitem__(self, idx):
        return self.S[idx], self.x[idx], self.t[idx]


def create_dataloader_real(params):
    """
    Creates a dataloader based on a dataset which is either ScatDataReal
    or ScatDataRealFromFile.
    params is a dictionary which should contain the following keys:
        -if using files, simply 'prefix_files' as expected by
         ScatDataRealFromFile
        -if recomputing all scatterings, then all the arguments expected
         by ScatDataReal, with their names
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
        sd = ScatDataRealFromFile(prefix_files=params['prefix_files'])
    else:
        sd = ScatDataReal(**params)
    dataloader = DataLoader(
        sd, batch_size=params['batch_size'], num_workers=params['num_workers'],
        drop_last=True, shuffle=False)
    return dataloader


def create_dataloader_list_real(params):
    # in this case, we may assume that 'num_split' is in params, and != None
    # so we may do:
    dataloader_list = []
    for i in range(params['num_split']):
        other_params = params.copy()
        other_params['id_split'] = i
        dataloader_list.append(create_dataloader_real(other_params))
    return dataloader_list


def meta_to_dic(meta, t_start, dataset_name='timit'):
    if dataset_name == 'timit':
        dic = {'region': meta[0][0],
               'speaker': meta[1][0],
               'recording': meta[2][0],
               'is_noisy': meta[3][0],
               'cut': t_start}
    elif dataset_name == 'vctk':
        dic = {'region': meta[0][0],
               'speaker': meta[1][0],
               'is_noisy': meta[2][0],
               'cut': t_start}
    elif dataset_name == 'nsynth':
        dic = {'instrument': meta[0][0],
               'velocity': meta[1][0],
               'pitch': meta[2][0],
               'is_noisy': meta[3][0],
               'cut': t_start}
    else:
        raise ValueError('Unknown dataset :' + str(dataset_name))
    return dic


def create_dataset(params):
    """
    Creates a real dataset from the raw signals.
    It proceeds in 2 steps:
    1/ Iteration over the dataset (possibly in parallel thanks to DataLoader)
        to compute all scatterings, using ScatDataReal
    2/ Cutting the resulting signals at the adequate size, and keeping the meta
        information concerning this creation.
    Then, the resulting dataset is saved.

    params is a dictionary containing the keys for all subsequent calls:
        - create_dataloader_real
        - step 2/, which involves params['seq_len_S'], which is the maximal
        temporal extent which is accepted for scattering vectors.
        The signals are kept accordingly.
        There might be an overlap between some samples at the last value,
        in order to prevent losing some parts of the signal.
        - 'path_save': folder (with / at the end)
        - 'timestamp': identifier for the dataset (name)
    """
    dataname = params['dataset_name']
    # create the dataloader
    # Only 1 dataloader in fact!!
    dataloader = create_dataloader_real(params)
    # NB: for VCTK, it should contain an 'id split'
    # Precomputing all the scatterings in parallel:
    # we simply hack the dataloader multiprocessing capabilities
    S_acc, x_acc, t_acc, meta_acc = [], [], [], []
    n_iter = 0
    try:
        for samples in tqdm(dataloader, desc="Scat. comput."):
            S, x, t, meta = samples
            S_acc.append(S)
            x_acc.append(x)
            t_acc.append(t)
            meta_acc.append(meta)
            n_iter += 1
    except:
        print('There was an error somewhere!')
        assert len(S_acc) == len(dataloader)
    # cut at the right size
    print('Postprocessing...')
    S_all = []
    x_all = []
    t_all = []
    meta_all = []
    seq_len_S = params['seq_len_S']
    for i in range(len(S_acc)):
        if meta_acc[i][-1][0]:  # if this is an example to keep afterwards
            k = S_acc[i].size(-1) // seq_len_S
            if k >= 1:  # otherwise, we do not consider this sample!
                for l in range(k):
                    S_temp = S_acc[i][...,
                                      l * seq_len_S:(l + 1) * seq_len_S].clone()
                    t_temp = t_acc[i][0, l * seq_len_S:(l + 1) * seq_len_S].clone()
                    t_start = t_temp[0]
                    x_temp = x_acc[i][..., t_temp[0]: t_temp[-1] + 1].clone()
                    t_temp -= t_start
                    dic = meta_to_dic(meta_acc[i], t_start,
                                      dataset_name=dataname)
                    S_all.append(S_temp)
                    t_all.append(t_temp.view(1, -1))
                    x_all.append(x_temp)
                    meta_all.append(dic)
                if S_acc[i].size(-1) % seq_len_S != 0:
                    # we take the last one
                    S_temp = S_acc[i][..., -seq_len_S:].clone()
                    t_temp = t_acc[i][0, -seq_len_S:].clone()
                    t_start = t_temp[0]
                    x_temp = x_acc[i][..., t_temp[0]: t_temp[-1] + 1].clone()
                    t_temp -= t_start
                    dic = meta_to_dic(meta_acc[i], t_start,
                                      dataset_name=dataname)
                    S_all.append(S_temp)
                    t_all.append(t_temp.view(1, -1))
                    x_all.append(x_temp)
                    meta_all.append(dic)
    # Concatenate
    S = torch.cat(S_all, dim=0).contiguous()
    x = torch.cat(x_all, dim=0).contiguous()
    t = torch.cat(t_all, dim=0).contiguous()
    # Display
    print('Final number of examples =', S.size(0))
    # make a final permutation to make sure that the examples will not always
    # come from the same example
    if 'shuffle' in params:
        shuffle = params['shuffle']
    else:
        shuffle = True
    if shuffle:
        perm = np.random.permutation(S.size(0))
        perm_th = torch.from_numpy(perm).long()
        S = S[perm_th]
        x = x[perm_th]
        t = t[perm_th]
        meta_new = [meta_all[perm[i]] for i in range(len(meta_all))]
    else:
        meta_new = meta_all
    # save
    print('Saving...')
    path_save = params['path_save'] + params['timestamp']
    if 'id_split' in params:  # for VCTK
        path_save = path_save + '_' + str(params['id_split'])
    torch.save(S, path_save + '_scatterings.pth')
    torch.save(x, path_save + '_signals.pth')
    torch.save(t, path_save + '_times.pth')
    with open(path_save + '_meta.json', 'w') as f:
        json.dump(meta_new, f)
    with open(path_save + '_params.json', 'w') as f:
        json.dump(params, f)
    return params['timestamp']


def merge_datasets(params, delete=False):
    """
    This function merges the different datasets produced independently
    (useful when a large dataset returns bugs and needs to be processed
    in multiple chunks)
    """
    S_acc, x_acc, t_acc, meta_acc = [], [], [], []
    for id_split in range(params['num_split']):
        path_save = params['path_save'] + params['timestamp'] +\
            '_' + str(id_split)
        S_acc.append(torch.load(path_save + '_scatterings.pth'))
        x_acc.append(torch.load(path_save + '_signals.pth'))
        t_acc.append(torch.load(path_save + '_times.pth'))
        with open(path_save + '_meta.json', 'r') as f:
            meta_temp = json.load(f)
        meta_acc = meta_acc + meta_temp  # simple concatenation
        # Checks
        assert S_acc[-1].size(0) == x_acc[-1].size(0)
        assert S_acc[-1].size(0) == t_acc[-1].size(0)
        assert S_acc[-1].size(0) == len(meta_temp)
        print('id_split', id_split, '| found', S_acc[-1].size(0),
              'new examples')
        # remove these files if necessary
        if delete:
            list_suffix = ['_scatterings.pth', '_signals.pth', '_times.pth',
                           '_meta.json']  # leave the _params.json, as a trace
            for suffix in list_suffix:
                os.remove(path_save + suffix)
    # Concatenate the tensors
    S = torch.cat(S_acc, dim=0)
    x = torch.cat(x_acc, dim=0)
    t = torch.cat(t_acc, dim=0)
    # Display
    print('\nFinal number of examples =', S.size(0))

    # Reshuffle them just to make sure
    perm = np.random.permutation(S.size(0))
    perm_th = torch.from_numpy(perm)
    S = S[perm_th]
    x = x[perm_th]
    t = t[perm_th]
    meta_new = [meta_acc[perm[i]] for i in range(len(meta_acc))]

    # Save them
    print('Saving...')
    path_save = params['path_save'] + params['timestamp']
    torch.save(S, path_save + '_scatterings.pth')
    torch.save(x, path_save + '_signals.pth')
    torch.save(t, path_save + '_times.pth')
    with open(path_save + '_meta.json', 'w') as f:
        json.dump(meta_new, f)
    with open(path_save + '_params.json', 'w') as f:
        json.dump(params, f)
    return params['timestamp']
