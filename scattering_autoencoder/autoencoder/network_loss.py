import numpy as np
import time
import torch
from kymatio import Scattering1D
from scattering_autoencoder.utils import mmd_linear_th
import pdb

class LossStorage(object):
    """
    This class stores the losses throughout the iterations
    """
    def __init__(self, loss_type='scat', num_batches=5, num_epochs=10,
                 **kwargs):
        super(LossStorage, self).__init__()
        self.loss_type = loss_type
        self.dic_loss = self._create_dic_loss(loss_type)
        self.num_batches = num_batches
        self.num_epochs = num_epochs
        # pre-assign the accumulators
        self.acc_loss_batch = {}
        self.acc_loss = {}
        self.size_acc = {}  # to be filled at the first iteration
        self.time_start_epoch = 0
        # flag for the first epoch, which is special
        self.is_first_epoch = True

    def _create_dic_loss(self, loss_type):
        if loss_type == 'pyscat':
            dic_loss = ['scattering', 'all']
        elif loss_type == 'pyscat_mmd':
            dic_loss = ['scattering', 'all', 'mmd']
        else:
            dic_loss = ['all']
        return dic_loss

    def init_epoch(self, n_epoch):
        if n_epoch == 0:
            self.acc_loss_batch = {k: [] for k in self.dic_loss}
            self.is_first_epoch = True
        else:
            self.acc_loss_batch = {
                k: np.squeeze(np.zeros([self.num_batches, self.size_acc[k]]))
                for k in self.dic_loss}
            self.is_first_epoch = False
        self.time_start_epoch = time.time()

    def record_batch(self, temp_loss, iter_batch=0):
        # update the keys of the dictionary
        self.dic_loss = sorted(list(temp_loss.keys()))
        for k in self.dic_loss:
            if self.is_first_epoch:
                self.acc_loss_batch[k].append(temp_loss[k])
            else:
                self.acc_loss_batch[k][iter_batch] = temp_loss[k]

    def display_loss(self, n_epoch):
        mean_loss = np.array(self.acc_loss_batch['all']).mean()
        tic = self.time_start_epoch
        toc = time.time()
        print('\tEpoch', n_epoch, 'mean loss =', mean_loss,
              'done in', toc - tic, 's')

    def record_epoch(self, n_epoch):
        if n_epoch == 0:
            # dynamically measure the size of each key
            self.size_acc = {}
            for k in self.dic_loss:
                if isinstance(self.acc_loss_batch[k][0], np.ndarray):
                    self.size_acc[k] = self.acc_loss_batch[k][0].size
                else:  # float
                    self.size_acc[k] = 1

            # dynamically allocate the size of the main accumulator
            self.acc_loss = {
                k: np.squeeze(np.zeros([self.num_epochs * self.num_batches,
                                        self.size_acc[k]]))
                for k in self.dic_loss
            }

            # array-ify the accumulator across batches
            for k in self.dic_loss:
                self.acc_loss_batch[k] = np.squeeze(
                    np.array(self.acc_loss_batch[k]))
        indices = np.arange(n_epoch * self.num_batches,
                            (n_epoch + 1) * self.num_batches,
                            dtype=int)
        for k in self.dic_loss:
            self.acc_loss[k][indices] = self.acc_loss_batch[k]

    def get_loss(self):
        return self.acc_loss


def precompute_normalization(responses, p=2):
    """
    Computes the mean across samples (dim=0) and time (dim=-1)
    for all channels
    """
    if p == 2:
        norms = torch.sqrt(torch.mean(responses**2, dim=0).mean(dim=-1))
    elif p == 1:
        norms = torch.mean(torch.abs(responses), dim=0).mean(dim=-1)
    return norms.view(-1, 1).contiguous()


def loss_sub(x_gen_sub, x_true_sub, norm_sub, perceptual=False, eps=1e-3,
             p=2):
    diff_sub_raw = x_gen_sub - x_true_sub
    diff_sub_norm = diff_sub_raw / norm_sub
    if perceptual:
        denom_sub = torch.abs(x_true_sub / norm_sub) + eps
        diff_sub_norm = diff_sub_norm / denom_sub
    if p == 2:
        diff_p = diff_sub_norm**2
    elif p == 1:
        diff_p = torch.abs(diff_sub_norm)
    ratio_sub = torch.mean(diff_p, dim=0).mean(dim=-1)
    loss_sub = torch.mean(ratio_sub)
    return loss_sub, ratio_sub


class LossComputer(object):
    """
    Some sort of criterion, but ad hoc.
    It returns a Variable coming from a true graph
    """
    def __init__(self, loss_type='scat', J_loss=2, Q_loss=2,
                 xi_max_loss=0.25, normalization_loss='l1',
                 include_poly_moments_0=True, average_U1=False,
                 size_domain=256, is_cuda=True,
                 whole_dataset_cuda=True, batch_size=128, eps=1e-6,
                 apply_normalization=True, L=6, xi_min_loss=0.04,
                 criterion_amplitude=1e-3, joint_S1=False, joint_U1=False,
                 L_joint=3, joint_U2=False, order2=False, backend='cufft',
                 l1_regularization_order1=False, mu_order1=1e-1, p_order=None,
                 perceptual=False, subsample_factor=1, oversampling=0,
                 **kwargs):
        self.loss_type = loss_type
        self.is_cuda = is_cuda
        self.whole_dataset_cuda = whole_dataset_cuda
        self.norm_factors = {}  # pre_assignment
        self.batch_size = batch_size  # at runtime, not for pre-computing
        self.eps = eps  # for numerical stability
        self.apply_normalization = apply_normalization
        if (loss_type == 'pyscat') or (loss_type == 'pyscat_mmd'):
            target_type = torch.cuda.FloatTensor if is_cuda\
                else torch.FloatTensor
            max_order = 2 if order2 else 1
            self.scatterer = Scattering1D(
                J_loss, size_domain, Q=Q_loss, max_order=max_order,
                average=average_U1, vectorize=True, oversampling=oversampling)
            if is_cuda:
                self.scatterer = self.scatterer.cuda()
            self.p_order = {'scattering': 2} if p_order is None else p_order
        elif loss_type == 'mse':
            # nothing to do here
            pass
        else:
            raise ValueError('Unknown loss type ' + str(loss_type))

    def precompute_target(self, x_true):
        if (self.loss_type == 'pyscat') or (self.loss_type == 'pyscat_mmd'):
            x_true_scat = self._precompute_response_pyscat(x_true)
            if self.apply_normalization:
                norm_scat = precompute_normalization(
                    x_true_scat, p=self.p_order['scattering']) + self.eps
            else:
                norm_scat = \
                    x_true_scat.new(x_true_scat.size(1)).fill_(1.).detach().requires_grad_(False)
            if self.is_cuda:
                norm_scat = norm_scat.cuda()
            self.norm_factors = {'scattering': norm_scat}
            return {'scattering': x_true_scat}
        elif self.loss_type == 'mse':
            # no need to precompute the response
            norm_x = precompute_normalization(x_true) + self.eps
            self.norm_factors = {'x': norm_x.detach().cuda()}
            return {'x': x_true}  # encapsulation for backward compatibility
        else:
            raise ValueError('Unknown loss type ' + str(self.loss_type))

    def _precompute_response_pyscat(self, x_true):  # only for x_true!
        acc_resp = []
        num_batches = x_true.size(0) // self.batch_size
        for i in range(num_batches):
            source = x_true[i * self.batch_size: (i + 1) * self.batch_size]
            already_cuda = True
            if not(source.is_cuda) and self.is_cuda:
                already_cuda = False
                source = source.cuda()  # move it to the GPU
            source_scat = self.scatterer.forward(
                source.view(source.shape[0], source.shape[-1])
            )
            if not(already_cuda):  # the dataset was not stored on GPU
                source_scat = source_scat.cpu()
            acc_resp.append(source_scat)
        x_true_resp = torch.cat(acc_resp, dim=0).detach().requires_grad_(False)
        return x_true_resp

    def compute_loss(self, x_gen, x_true):
        if (self.loss_type == 'pyscat') or (self.loss_type == 'pyscat_mmd'):
            return self._compute_loss_pyscat(x_gen, x_true)
        elif self.loss_type == 'mse':
            return self._compute_loss_mse(x_gen, x_true)
        else:
            raise ValueError('Unknown loss type ' + str(self.loss_type))

    def _compute_loss_pyscat(self, x_gen, x_true):
        x_gen_scat = self.scatterer.forward(
            x_gen.view(x_gen.shape[0], x_gen.shape[-1])
        )
        # compute loss
        loss, ratio = loss_sub(x_gen_scat, x_true['scattering'],
                               self.norm_factors['scattering'],
                               p=self.p_order['scattering'])
        temp_loss = {'all': loss.cpu().detach().item(),
                     'scattering': ratio.cpu().detach().numpy()}
        return loss, temp_loss

    def _compute_loss_mse(self, x_gen, x_true):
        diff = x_true['x'] - x_gen
        loss = torch.mean((diff / self.norm_factors['x'])**2)
        temp_loss = {'all': loss.cpu().detach().item()}
        return loss, temp_loss


class LossMMD(object):
    def __init__(self, size_domain, J_mmd=6, Q_mmd=12, order2_mmd=False,
                 normalize_mmd='l2', criterion_amplitude_mmd=1e-2,
                 batch_size=128, eps=1e-5, is_cuda=True, order0_mmd=True,
                 loss_type_mmd='static', **kwargs):
        """
        Loss types: "static", "dynamic" (equality in distribution with respect
            to actual scatterings, not just the mean)
        """
        max_order = 2 if order2_mmd else 1
        self.scatterer = Scattering1D(
            J_mmd, size_domain, Q=Q_mmd, max_order=max_order, average=average_U1,
            oversampling=0, vectorize=True)
        if is_cuda:
            self.scatterer = self.scatterer.cuda()
        self.batch_size = batch_size
        self.eps = eps
        self.is_cuda = is_cuda
        self.include_order0 = order0_mmd
        self.loss_type = loss_type_mmd
        self.norm_factors = None

    def _compute_scat_batches(self, x_true):
        # apply the scatterer in batches
        acc_scat = []
        num_batches = x_true.size(0) // self.batch_size
        for i in range(num_batches):
            source = x_true[i * self.batch_size: (i + 1) * self.batch_size]
            already_cuda = True
            if not(source.is_cuda) and self.is_cuda:
                already_cuda = False
                source = source.cuda()  # move it to the GPU
            source_scat = self.scatterer.forward(
                source.view(source.shape[0], source.shape[-1])
            )
            if not(already_cuda):
                source = source.cpu()  # put back the tensor on CPU
            acc_scat.append(source_scat)
        scat_all = torch.cat(acc_scat, dim=0)
        if not(self.include_order0):
            scat_all = scat_all[:, 1:]
            scat_all = scat_all.contiguous()
        return scat_all

    def _precompute_target_static(self, x_true):
        scat_all = self._compute_scat_batches(x_true)
        # precompute the normalization
        self.norm_factors = torch.squeeze(
            precompute_normalization(scat_all).detach())
        # at this point, scat_all is a GPU tensor (if cuda is active)
        # compute the mean across time and batch
        mean_scat = torch.mean(scat_all, dim=-1).mean(dim=0)
        # make it a new, independent variable of adequate size
        mean_scat = mean_scat.detach().contiguous().requires_grad_(False)
        return mean_scat

    def _precompute_target_dynamic(self, x_true):
        scat_all = self._compute_scat_batches(x_true)
        self.norm_factors = torch.abs(scat_all).mean(dim=0).mean(
                dim=-1).contiguous().view(-1, 1).contiguous().detach().requires_grad_(False)
        # renormalize scat_all
        scat_all = scat_all / self.norm_factors
        # reshape it to collapse all last dimensions
        scat_all = scat_all.view(scat_all.size(0),
                                 scat_all.size(1) * scat_all.size(2))
        # Detach the variable
        scat_all = scat_all.detach().requires_grad_(False)

        return scat_all

    def _precompute_target_marginal(self, x_true):
        scat_all = self._compute_scat_batches(x_true)
        self.norm_factors = \
            torch.abs(scat_all).mean(dim=0).mean(
                dim=-1).contiguous().view(-1, 1).contiguous().detach().requires_grad_(False)
        # renormalize scat_all
        scat_all = scat_all / self.norm_factors
        # detache scat_all
        scat_all = scat_all.detach().requires_grad_(False)
        return scat_all

    def precompute_target(self, x_true):
        if self.loss_type == 'static':
            return self._precompute_target_static(x_true)
        elif self.loss_type == 'dynamic':
            return self._precompute_target_dynamic(x_true)
        elif self.loss_type == 'marginal':
            return self._precompute_target_marginal(x_true)
        else:
            raise TypeError('Unknown loss type mmd ' + str(self.loss_type))

    def _compute_mmd_static(self, x_gen, mu):
        # get the scattering
        scat_gen = self.scatterer.forward(
            x_gen.view(x_gen.shape[0], x_gen.shape[-1])
        )
        if not(self.include_order0):
            scat_gen = scat_gen[:, 1:]  # remove order 0
        assert scat_gen.size(1) == mu.size(0)
        # compute the average scattering across time and batch
        average_scat = torch.mean(scat_gen, dim=0).mean(dim=-1)
        # compute the relative ratio
        ratios = (average_scat - mu) / self.norm_factors
        # get the mean across coordinates as well
        loss_mmd = torch.mean(ratios**2)
        # save the ratios
        temp_loss = {'mmd': np.abs(ratios.detach().cpu().numpy())}
        return loss_mmd, temp_loss

    def _compute_mmd_dynamic(self, x_gen, scat_true):
        # compute the scattering
        scat_gen_raw = self.scatterer.forward(
            x_gen.view(x_gen.shape[0], x_gen.shape[-1])
        )
        if not(self.include_order0):
            scat_gen_raw = scat_gen_raw[:, 1:]
            scat_gen_raw = scat_gen_raw.contiguous()
        # renormalize
        scat_gen = scat_gen_raw / self.norm_factors
        # collapse all dimensions
        scat_gen = scat_gen.view(scat_gen.size(0),
                                 scat_gen.size(1) * scat_gen.size(2))
        # compute the mmd
        loss_mmd = mmd_linear_th(scat_gen, scat_true)
        return loss_mmd, {'mmd': loss_mmd.data.cpu().numpy()}

    def _compute_mmd_marginal(self, x_gen, scat_true):
        # compute the scattering
        scat_gen_raw = self.scatterer.forward(
            x_gen.view(x_gen.shape[0], x_gen.shape[-1])
        )
        if not(self.include_order0):
            scat_gen_raw = scat_gen_raw[:, 1:]
        # renormalize the scattering
        scat_gen = scat_gen_raw / self.norm_factors
        # compute the marginal across coordinates
        coords_true = scat_true.mean(dim=-1)
        coords_gen = scat_gen.mean(dim=-1)
        loss_mmd = mmd_linear_th(coords_gen, coords_true)
        ratios = [loss_mmd.data.cpu().numpy()]
        # compute the marginal across time
        for i in range(scat_gen.size(1)):
            current = mmd_linear_th(scat_gen[:, i, :], scat_true[:, i, :])
            loss_mmd += current
            ratios.append(current.data.cpu().numpy())
        return loss_mmd, {'mmd': np.squeeze(np.array(ratios))}

    def compute_mmd(self, x_gen, x_true):
        if self.loss_type == 'static':
            # in this case, x_true is just the "mu", a mean over all
            # times and batches
            return self._compute_mmd_static(x_gen, x_true)
        elif self.loss_type == 'dynamic':
            # in this case, x_true is a vector of scatterings, with
            # collapsed dimensions time and channels
            return self._compute_mmd_dynamic(x_gen, x_true)
        elif self.loss_type == 'marginal':
            return self._compute_mmd_marginal(x_gen, x_true)
        else:
            raise TypeError('Unknown loss type mmd ' + str(self.loss_type))
