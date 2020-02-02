from argparse import ArgumentParser

import os
from scattering_autoencoder.utils import get_timestamp
from scattering_autoencoder.datasets import ScatDataMusic, create_dataset_music

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_data", type=str, help="npy file containing the dataset")
    parser.add_argument("--path_save", type=str, help="Root folder where to save the preprocessed dataset")
    args = parser.parse_args()

    params_train = {
        'J': 10,
        'Q': 12,
        'N': 3,
        'include_haar': False,
        'joint': False,
        'len_fft_joint': 0,
        'avg_U1_before_joint': True,
        't_send_factor': 3,
        'max_length': int(2e5),
        'seq_len_S': 16,
        'normalize_amplitude': True,
        'path_dataset': args.path_data,
        'path_save': os.path.join(args.path_save, 'data/music_preprocess/train/'),
        'batch_size': 1,
        'num_workers': 1,
        'timestamp': get_timestamp(),
        'file_responsible': 'preprocess_dataset.py'
    }

    params_train['dx'] = 2 ** params_train['J']

    params_test = params_train.copy()
    params_test['path_dataset'] = params_test['path_dataset'].replace('train', 'test')
    params_test['path_save'] = params_test['path_save'].replace('train', 'test')

    print(params_train["timestamp"])
    if not os.path.exists(params_train["path_save"]):
        os.makedirs(params_train["path_save"])
    create_dataset_music(params_train)

    print(params_test['timestamp'])
    if not os.path.exists(params_test["path_save"]):
        os.makedirs(params_test["path_save"])
    create_dataset_music(params_test)
