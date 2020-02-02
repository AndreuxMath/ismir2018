import time
from itertools import chain
import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm
import json
import numpy as np
import pdb
from tqdm import tqdm
from .network_core import compute_time_convgen
from .network_loss import LossStorage, LossComputer
from .network_loss import LossMMD
from .scattering_generator import create_generator_SynthScat


def compute_covariance(X):
    mean_X = np.mean(X, axis=0)
    X_centered = X - mean_X
    cov_X = np.dot(X_centered.T, X_centered) / float(X_centered.shape[0])
    return cov_X


def normalize_amplitudes(S, params):
    # Assumes that S is the tensor directly taken from hack_dataloader
    S = S.detach().cpu().requires_grad_(False)
    # compute the average amplitude for non-zero scattering
    ampl = torch.mean(torch.abs(S[:, 1:, :]), dim=0).mean(dim=-1)
    # Create a new tensor
    S_new = S.new(S.size())
    S_new[:, 0] = S[:, 0]
    S_new[:, 1:] = S[:, 1:] / ampl.view(-1, 1)
    assert np.any(S_new[:, 1:].numpy() < 0) == False
    # Create a new tensor
    if params['is_cuda'] and params['whole_dataset_cuda']:
        S_new = S_new.cuda()
    # remake it differentiable
    S_new = S_new.detach().requires_grad_()
    return S_new, ampl


def train_predictor_S(s_predictor, dataloader, optimizer, criterion, params,
                      scheduler=None):
    S_all, _ = hack_dataloader(dataloader, params)
    if 'normalize_amplitudes' in params:
        if params['normalize_amplitudes']:
            S_all, ampl = normalize_amplitudes(S_all, params)
        else:
            ampl = torch.ones(S_all.size(1) - 1)
    else:
        ampl = torch.ones(S_all.size(1) - 1)

    bs = params['batch_size']
    num_batches = S_all.size(0) // bs
    acc_loss = []
    # Train the model
    for n_epoch in range(params['num_epochs']):
        acc_loss2 = []
        tic = time.time()
        for i in range(num_batches):
            s_predictor.zero_grad()
            S_J = S_all[i * bs: (i + 1) * bs]
            # make them variables
            S_source = S_J[:, :, :-1]  # B x C x T - 1
            # predict them
            S_predict = s_predictor.forward(S_source)
            # get the target of adequate size
            S_target = S_J[:, :, -S_predict.size(-1):]
            # compute the loss
            mean_amplitude = torch.mean(S_target**2, dim=0).mean(dim=-1)
            diff = torch.mean((S_target - S_predict)**2, dim=0).mean(dim=-1)
            rel_diff = torch.mean(diff / mean_amplitude)
            loss = rel_diff
            temp_loss = loss.data.cpu().numpy()[0]
            acc_loss2.append(temp_loss)
            # make one optimization step
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        mean_loss = np.array(acc_loss2).mean()
        print('\tEpoch', n_epoch, 'mean loss =', mean_loss, 'done in',
              time.time() - tic, 's')
        acc_loss = acc_loss + acc_loss2
    print('Computing final Sigma')
    # Measure the \Sigma and not sigmas
    S_source_all = S_all[:, :, :-1]
    S_predict_all = s_predictor.forward(S_source_all)
    S_target_all = S_all[:, :, -S_predict_all.size(-1):]
    diff_all = (S_predict_all - S_target_all).data.cpu().numpy()
    diff_all = diff_all.transpose(0, 2, 1)
    diff_all = diff_all.reshape(-1, diff_all.shape[-1])

    cov_error = compute_covariance(diff_all)
    U, singular_vals, V = np.linalg.svd(cov_error)
    root_singular_vals = np.sqrt(singular_vals)
    cov_error_sqrt = np.dot(np.dot(U, np.diag(root_singular_vals)), V)
    cov_error_sqrt_th = torch.from_numpy(cov_error_sqrt)
    # return the results
    return s_predictor, np.array(acc_loss), cov_error_sqrt_th, ampl


def hack_dataloader(dataloader, params):
    S_acc = []
    x_acc = []
    t_acc = []
    for samples in dataloader:
        S_J, x, times_J = samples
        S_acc.append(S_J)
        x_acc.append(x)
        t_acc.append(times_J)
    # concatenate
    S_all = torch.cat(S_acc, dim=0)
    x_all = torch.cat(x_acc, dim=0)
    t_all = torch.cat(t_acc, dim=0)
    # get the times for x0
    times_J = t_all[0].numpy()
    times = compute_time_convgen(
        times_J, params['J'], params['past_size'], params['look_ahead'])
    time_0 = times[0]
    # compute the target
    x_true = x_all[..., time_0[0]:time_0[-1] + 1].contiguous()
    # cut if necessary
    if 'maximal_size_dataset' in params:
        maximal_size_dataset = params['maximal_size_dataset']
        if maximal_size_dataset is not None:
            S_all = S_all[:maximal_size_dataset]
            x_true = x_true[:maximal_size_dataset]
    # keep only a sub number of coordinates if required
    if 'num_coords_to_keep' in params:
        num_coords_to_keep = params['num_coords_to_keep']
        S_all = S_all[:, :num_coords_to_keep]
    # move to cuda if necessary
    if params['is_cuda'] and params['whole_dataset_cuda']:
        S_all = S_all.cuda()
        x_true = x_true.cuda()
    # return it
    print("S_all size: ", S_all.size())
    print("x_true size: ", x_true.size())
    return S_all, x_true


def slice_dict_tensors(v, inds_to_take):
    return {k: v[k][inds_to_take] for k in v.keys()}


def send_cuda_dict_tensors(v):
    return {k: v[k].cuda() for k in v.keys()}


def check_no_nans(model):
    nonan = True
    for p in model.parameters():
        nonan = nonan and (np.any(np.isnan(p.data.numpy())) == False)
    return nonan


def save_model(model, params, n_epoch):
    tic = time.time()
    if params['is_cuda']:
        model = model.cpu()
    # Prior to saving it, check that there are no nans
    nonan = check_no_nans(model)
    if nonan:  # we save only if no nans, to avoid corrupting storage
        torch.save(model, params['prefix_save'] + '_conv_gen_temp.pth')
        with open(params['prefix_save'] + '_conv_gen_temp_meta.txt', 'w') as f:
            f.write(str(n_epoch))
        message = 'Saved model '
    else:
        message = 'Nans in model! No saving '
    if params['is_cuda']:
        model = model.cuda()
    toc = time.time()
    print('\t\t', message, 'in', toc - tic, 's')
    return model


def train_model(conv_gen, dataloader, optimizer, params, scheduler=None):
    bs = dataloader.batch_size
    num_epochs = params['num_epochs']
    save_every = params['save_every']
    S_J, x_true_raw = hack_dataloader(dataloader, params)
    must_put_cuda = not(params['whole_dataset_cuda']) and params['is_cuda']
    criterion = LossComputer(size_domain=x_true_raw.size(-1), **params)

    # precompute the target (if necessary)
    x_true = criterion.precompute_target(x_true_raw)
    if 'maximal_size_dataset_sub' in params:
        sub_size = params['maximal_size_dataset_sub']
        for k in x_true.keys():
            x_true[k] = x_true[k][:sub_size]
        S_J = S_J[:sub_size]
        must_put_cuda_new = (not(params['whole_dataset_cuda_sub']) and
            params['is_cuda'])
        if must_put_cuda and not(must_put_cuda_new):
            S_J = S_J.cuda()
            for k in x_true.keys():
                x_true[k] = x_true[k].cuda()
        must_put_cuda = must_put_cuda_new
    print('Final scattering', S_J.size())
    for k in x_true.keys():
        print('Final', k, x_true[k].size())
    
    # precompute the storage device
    num_samples = S_J.size(0)
    num_batches = num_samples // bs
    storage = LossStorage(loss_type=params['loss_type'],
                          num_batches=num_batches,
                          num_epochs=num_epochs)
    # preassign the flag for nans
    found_nan = False
    # Main training loop
    for n_epoch in tqdm(range(num_epochs)):
        storage.init_epoch(n_epoch)
        perm = torch.arange(num_samples).long()
        perm = perm.cuda() if not(must_put_cuda) and params["is_cuda"] else perm
        for i in range(num_batches):
            conv_gen.zero_grad()
            # slice dict_tensor
            x_true_batch = slice_dict_tensors(
                x_true, perm[i * bs: (i + 1) * bs])
            S_J_batch = S_J[perm[i * bs: (i + 1) * bs]]
            if must_put_cuda:
                x_true_batch = send_cuda_dict_tensors(x_true_batch)
                S_J_batch = S_J_batch.cuda()
            # Forward the convolutions
            x_gen = conv_gen.forward(S_J_batch)
            # Compute the loss
            loss, temp_loss = criterion.compute_loss(x_gen, x_true_batch)
            # check the presence of nans
            if np.any(np.isnan(loss.data.cpu().numpy())):
                found_nan = True
                print('NaN detected at batch', i, '! Stopping early')
                break
            # record the loss
            storage.record_batch(temp_loss, iter_batch=i)
            # make one step
            loss.backward()
            # clip gradients if necessary
            if 'clip_gradient' in params:
                clip_grad_norm(conv_gen.parameters(), params['clip_gradient'],
                               norm_type=params['clip_gradient_norm'])
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if found_nan:
            break
        storage.record_epoch(n_epoch)
        if n_epoch % save_every == 0:
            storage.display_loss(n_epoch)
            conv_gen = save_model(conv_gen, params, n_epoch)
    return conv_gen, storage.get_loss()


def train_model_mmd(conv_gen, dataloader, optimizer, params, scheduler=None):
    lambda_rec_mmd = params['lambda_rec_mmd']
    bs = dataloader.batch_size
    num_epochs = params['num_epochs']
    save_every = params['save_every']
    S_J, x_true_raw = hack_dataloader(dataloader, params)
    must_put_cuda = not(params['whole_dataset_cuda']) and params['is_cuda']
    criterion = LossComputer(size_domain=x_true_raw.size(-1), **params)
    # precompute the target (if necessary)
    x_true = criterion.precompute_target(x_true_raw)
    if 'maximal_size_dataset_sub' in params:
        sub_size = params['maximal_size_dataset_sub']
        for k in x_true.keys():
            x_true[k] = x_true[k][:sub_size]
        S_J = S_J[:sub_size]
        must_put_cuda_new = (not(params['whole_dataset_cuda_sub']) and
                             params['is_cuda'])
        if must_put_cuda and not(must_put_cuda_new):
            S_J = S_J.cuda()
            for k in x_true.keys():
                x_true[k] = x_true[k].cuda()
        must_put_cuda = must_put_cuda_new
    print('Final scattering', S_J.size())
    for k in x_true.keys():
        print('Final', k, x_true[k].size())
    # Precompute the mean for the MMD
    criterion_mmd = LossMMD(size_domain=x_true_raw.size(-1), **params)
    x_true_for_mmd = criterion_mmd.precompute_target(x_true_raw)
    predicate2 = not(params['whole_dataset_cuda'])\
        and params['is_cuda'] and (params['loss_type_mmd'] != 'static')
    if predicate2:
        x_true_for_mmd = x_true_for_mmd.cpu()
    predicate = ('maximal_size_dataset_sub' in params) and\
        ((params['loss_type_mmd'] == 'dynamic') or
            (params['loss_type_mmd'] == 'marginal'))
    if predicate:
        sub_size = params['maximal_size_dataset_sub']
        x_true_for_mmd = x_true_for_mmd[:sub_size]
        must_put_cuda_new = (not(params['whole_dataset_cuda_sub']) and
                             params['is_cuda'])
        if must_put_cuda and not(must_put_cuda_new):
            x_true_for_mmd = x_true_for_mmd.cuda()
    print('Final size for x_mmd =', x_true_for_mmd.size())
    print('MMD loss defined')
    # precompute the storage device
    num_samples = S_J.size(0)
    num_batches = num_samples // bs
    storage = LossStorage(loss_type=params['loss_type'],
                          num_batches=num_batches,
                          num_epochs=num_epochs)
    # define the MMD generator
    with open(params['prefix_gen'] + '_params.json', 'r') as f:
        params_scatpred = json.load(f)
    assert params_scatpred['prefix_files'] == params['prefix_files']
    scatpred = torch.load(params['prefix_gen'] + '_scatpred.pth')
    M = torch.load(params['prefix_gen'] + '_cov_error_sqrt_th.pth')
    alpha = torch.load(params['prefix_gen'] + '_amplitudes.pth')
    scatgen = create_generator_SynthScat(
        scatpred, M, alpha, max_gen=params['max_gen'],
        past_size=params_scatpred['past_size'], dim_scattering=M.size(0),
        batch_size=params['batch_size'], num_workers=params['num_workers_mmd'],
        queue_size=params['queue_size_mmd'], length_return=S_J.size(-1),
        take_relu=params['take_relu'])
    print('MMD generator defined')
    # preassign the flag for nans
    found_nan = False
    # Main training loop
    for n_epoch in range(num_epochs):
        storage.init_epoch(n_epoch)
        perm = torch.randperm(num_samples).long()
        perm = perm.cuda() if not(must_put_cuda) else perm

        perm2 = torch.randperm(num_samples).long()
        perm2 = perm2.cuda() if not(must_put_cuda) else perm2
        for i in range(num_batches):
            conv_gen.zero_grad()
            # slice dict_tensor for true loss
            x_true_batch = slice_dict_tensors(x_true,
                                              perm[i * bs: (i + 1) * bs])
            S_J_batch = S_J[perm[i * bs: (i + 1) * bs]]
            if must_put_cuda:
                x_true_batch = send_cuda_dict_tensors(x_true_batch)
                S_J_batch = S_J_batch.cuda()
            # get a new batch:
            synth_scat = scatgen.get()
            if params['is_cuda']:
                synth_scat = synth_scat.cuda()
            synth_scat = synth_scat
            # get x_true_for_mmd_batch
            if params['loss_type_mmd'] == 'static':
                x_true_for_mmd_batch = x_true_for_mmd
            else:
                x_true_for_mmd_batch =\
                    x_true_for_mmd[perm2[i * bs: (i + 1) * bs]]
                if must_put_cuda:
                    x_true_for_mmd_batch = x_true_for_mmd_batch.cuda()
            # Reconstruction error
            x_gen = conv_gen.forward(S_J_batch)
            loss_rec, temp_loss = criterion.compute_loss(x_gen, x_true_batch)
            # MMD error
            x_synth = conv_gen.forward(synth_scat)
            loss_mmd, temp_loss_mmd = criterion_mmd.compute_mmd(
                x_synth, x_true_for_mmd_batch)
            # Merge all errors
            loss = loss_rec + lambda_rec_mmd * loss_mmd
            temp_loss.update(temp_loss_mmd)
            # check the presence of nans
            if np.any(np.isnan(loss.data.cpu().numpy())):
                found_nan = True
                print('NaN detected at batch', i, '! Stopping early')
                break
            # make one step
            loss.backward()
            # record the loss
            storage.record_batch(temp_loss, iter_batch=i)
            # clip gradients if necessary
            if 'clip_gradient' in params:
                clip_grad_norm(conv_gen.parameters(), params['clip_gradient'],
                               norm_type=params['clip_gradient_norm'])
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if found_nan:
            break
        storage.display_loss(n_epoch)
        storage.record_epoch(n_epoch)
        if n_epoch % save_every == 0:
            conv_gen = save_model(conv_gen, params, n_epoch)
    return conv_gen, storage.get_loss()
