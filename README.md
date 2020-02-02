# Code
Paper "Music Generation and Transformation with Moment Matching-Scattering Inverse Networks", M. Andreux, S. Mallat, ISMIR 2018

# Audio samples
[Available on this page](http://andreuxmath.github.io/ismir.html)


# Setting up

* Install the conda environment: `conda env create -f conda-env.yml`
* Install the local package: `pip install -e .`

# Creating a dataset

## Toy dataset

Go to `scattering_autoencoder/datasets/toy_model_gen.py`, and call it with a `--path` argument specifying the place where the dataset will be stored and `--n_examples`, the number of examples in each train/test set.

## Music dataset

* Download some existing dataset here: train/test
* Compute their scattering with `experiments/music/preprocess_dataset.py`, with arguments:
** `--path_data`: path to the training data (`music_train_norm2_4096.npy`)
** `--path_save`: root folder where to save the preprocessed data.
** Note that this step can take up to 6 hours without CPU parallelization.
* Renormalize the channels using code from the notebook in `experiments/music/renormalize_datasets.ipynb`

# Launching experiments

The relevant file for training is `experiments/main.py`. It takes as argument
`--argfile`, which should be a json file containing parameters for the experiments. Examples thereof are provided in `experiments/music` and `experiments/toy`.

## Structure of the json files

* First block: "J" -> "dataset_name": arguments related to the scattering used as input to the convolutional network
* Second block: "loss_type" -> "p_order": arguments related to the loss function.
* Third block: "last_convolution" -> "size_channel_1": arguments related to the architecture of the network
* Fourth block: "lr" -> "gamma": arguments related to the training procedure
* Fifth block: "is_cuda" -> "timestamp_data": arguments related to the dataset. In particular, the argument "timestamp_data" should be updated according to the dataset which is used.
