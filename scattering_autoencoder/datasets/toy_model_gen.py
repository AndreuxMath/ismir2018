import numpy as np
from tqdm import tqdm
import torch
import os
import argparse
from scattering_autoencoder.scattering_recurrent import RecurrentScatteringNP
from scattering_autoencoder.utils import sample_omegas, get_timestamp


def generate_signal(size, n_examples, n_components=3):
    # frequencies
    omega_min = np.pi / 16
    omega_max = 3 * np.pi / 4
    omega = sample_omegas(
        num_components=n_components, num_examples=n_examples,
        omega_min=omega_min, omega_max=omega_max
    )

    # phases
    phase = np.random.rand(np.size(omega))
    phase = phase.reshape(np.shape(omega))
    phase *= 2 * np.pi

    # amplitudes
    ampl = np.random.rand(np.size(omega))
    ampl = ampl.reshape(np.shape(omega))
    ampl = (1 + 2 * ampl) / 3

    time = np.arange(size)
    cosines = np.cos(omega[:, :, None] * time[None, None, :] + phase[:, :, None])
    signal = np.sum(ampl[:, :, None] * cosines, axis=1)

    max_signal = np.max(signal, axis=1)
    signal = signal / max_signal[:, None]

    return signal, omega, phase, ampl / max_signal[:, None]


def get_generation_parameters_csv(freqs, phases, ampls):
    nb_cos = freqs.shape[1]
    headers = ["frequencies", "phases", "amplitudes"]
    csv_header = (', ' * nb_cos).join(headers)

    fpa = np.concatenate((freqs, phases, ampls), axis=1).astype(str)
    lines = [', '.join(fpa[i, :]) for i in range(fpa.shape[0])]
    data = '\n'.join(lines)

    csv = csv_header + data
    return csv


def smart_scatterer(x, J, scat, dump_transitory=True, init_skip=2):
    """Returns x (size T) and its scattering transform.
    If dump_transitory is true, the first 2 * 2**J timesteps are
    dumped in both x and its scattering.
    """

    # computes
    cs = 2 ** J  # chunksize
    assert x.shape[0] % cs == 0
    x_chunk = (x[cs * i:cs * (i + 1)] for i in range(x.shape[0] // cs))

    h = None  # memory initialization
    scat_transform = []
    for chunk in x_chunk:
        s_chunk, h = scat.forward(chunk, hidden_past=h,
                                  return_last_hidden=True)
        scat_transform.append(s_chunk[J][-1])  # subsampling
    scat_transform = np.stack(scat_transform, axis=1)
    # scat_transform has size (ScatSize, T)

    # dump transitory time
    if dump_transitory:
        scat_transform = scat_transform[:, init_skip:]
        x = x[init_skip * cs:]

    return x, scat_transform


def generate_data(size, n_examples, n_components,
                  J, Q, polynomial_order, init_skip=2):
    
    # generate data
    x, freqs, phases, ampls = generate_signal(size, n_examples, n_components=1)
    csv_descr = get_generation_parameters_csv(freqs, phases, ampls)
    x /= np.max(np.abs(x), axis=-1)[:, None]
    print("done generating signals")

    # get scattering transform
    scat = RecurrentScatteringNP(
        J, Q, N=polynomial_order, only_J=True, max_length=int(2e5)
    )

    compute_scat = [smart_scatterer(x[i], J, scat, init_skip)
                    for i in tqdm(range(x.shape[0]))]
    x, xwav = zip(*compute_scat)
    x = np.stack(x, axis=0)
    xwav = np.stack(xwav, axis=0)
    print("done computing scattering transforms")

    # renormalize
    xwav[:, 1:, :] = xwav[:, 1:, :] / np.mean(np.mean(np.abs(xwav[:, 1:, :]), axis=0), axis=-1).reshape(-1, 1)

    # convert to torch format
    x_torch = torch.FloatTensor(x).unsqueeze(1)
    xwav_torch = torch.FloatTensor(xwav)

    time = np.arange(xwav_torch.size(-1)) * 2 ** J
    time = np.tile(time, (np.shape(x)[0], 1))
    time = torch.LongTensor(time)

    return x_torch, xwav_torch, time, csv_descr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./", help="Path where the dataset will be created")
    parser.add_argument("--n_examples", type=int, default=16, help="Number of examples to create")
    args = parser.parse_args()

    # data complexity info
    n_components = 1  # number of frequencies for each signal

    # time info
    size = 13 * 1024
    n_examples = args.n_examples

    # scattering transform info
    J = 10
    Q = 12
    polynomial_order = 3

    # thresholding for initial transitory time
    init_skip = 2  # skip in time: init_skip * 2 ** J

    # generate train and test sets
    x_tr, xwav_tr, time_tr, csv_tr = generate_data(
        size, n_examples, n_components, J, Q, polynomial_order, init_skip)
    x_te, xwav_te, time_te, csv_te = generate_data(
        size, n_examples, n_components, J, Q, polynomial_order, init_skip)

    # mark with same timestamp
    timestamp = get_timestamp()

    # save train set
    savedir = os.path.join(args.path, 'data/toy_preprocess/train')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    torch.save(x_tr, os.path.join(savedir, timestamp + "_signals.pth"))
    torch.save(xwav_tr, os.path.join(savedir, timestamp + "_scatterings.pth"))
    torch.save(time_tr, os.path.join(savedir, timestamp + "_times.pth"))
    descr_path = os.path.join(savedir, timestamp + "_gendescr.csv")
    with open(descr_path, 'w') as descr_file:
        descr_file.write(csv_tr)

    # save test set
    savedir = os.path.join(args.path, 'data/toy_preprocess/test')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    torch.save(x_te, os.path.join(savedir, timestamp + "_signals.pth"))
    torch.save(xwav_tr, os.path.join(savedir, timestamp + "_scatterings.pth"))
    torch.save(time_te, os.path.join(savedir, timestamp + "_times.pth"))
    descr_path = os.path.join(savedir, timestamp + "_gendescr.csv")
    with open(descr_path, 'w') as descr_file:
        descr_file.write(csv_te)

    # print timestamp
    print("Saved with timestamp: ", timestamp)
