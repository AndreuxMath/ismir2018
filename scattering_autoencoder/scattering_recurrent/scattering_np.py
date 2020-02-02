"""
Numpy + numba version of these computations.
"""
import numpy as np
from numba import jit
from .utils_gammatones import (create_one_tuples,
                                build_wavelet_family,
                                create_psi1m, create_meta2)
import pdb


@jit(nopython=True)
def filter_add_inplace_np(input_seq, add_factor, h0=None):
    """
    Performs the recursive filtering:
    h[t] = h[t - 1] + add_factor * (input_seq[t] - add_factor[t - 1])
    with initial value h0 for h[-1] (first value)

    This is done inplace, so h is actually found back in input_seq

    INPUTS:
    - input_seq should be a numpy array of shape T x whatever
    - h0 should be a numpy array of shape whatever
    - add_factor can be a numpy array of shape 1 (or even larger, but then
        the other arrays must be prepared for the broadcast), or a float

    OUTPUTS
    nothing
    """
    if h0 is None:
        input_seq[0] = add_factor * input_seq[0]
    else:
        input_seq[0] = h0 + add_factor * (input_seq[0] - h0)
    # rest of the recursion
    T = input_seq.shape[0]
    for t in range(1, T):
        input_seq[t] = input_seq[t - 1] + \
            add_factor * (input_seq[t] - input_seq[t - 1])


@jit(nopython=True)
def filter_grad1D_inplace_np(input_seq, h0=None):
    """
    Follows the operation:
    output_seq[t] = input_seq[t] - input_seq[t - 1]
    This is performed inplace.
    We have output_seq[0] = input_seq[0] if h0 is None,
    otherwise output_seq[0] = input_seq[0] - h0
    """
    T = input_seq.shape[0]
    for t in range(T - 1, 0, -1):
        input_seq[t] = input_seq[t] - input_seq[t - 1]
    if h0 is not None:
        input_seq[0] = input_seq[0] - h0
    # Nothing to do then


def prepare_half_overlapping_small_case(input_array, Q):
    # this function handles the case where input_array.shape[-1] < Q
    missing = Q - input_array.shape[-1]
    pad_left = missing // 2
    pad_right = missing - pad_left
    pre_shape = input_array.shape[:-1] + (1,)
    inter_shape = pre_shape + (input_array.shape[-1],)
    pleft = np.zeros(pre_shape + (pad_left,), dtype=input_array.dtype)
    pright = np.zeros(pre_shape + (pad_right,), dtype=input_array.dtype)
    output_array = np.concatenate(
        [pleft, input_array.reshape(inter_shape), pright], axis=-1)
    return output_array


@jit(nopython=True)
def prepare_half_overlapping_large_case(input_array, Q):
    # this function handles the case where input_array.shape[-1] >= Q
    pre_shape = input_array.shape[:-1]
    half_Q = Q // 2
    K = input_array.shape[-1] // half_Q
    # determine the final shape depending on the divisibility of the shape
    if input_array.shape[-1] % half_Q == 0:
        num_windows = K - 1
        need_to_add_last = False
    else:
        num_windows = K
        need_to_add_last = True
    output_shape = pre_shape + (num_windows, Q)
    output_array = np.zeros(output_shape, dtype=input_array.dtype)
    for j in range(K - 1):
        target = input_array[...,
                             j * half_Q: j * half_Q + Q]
        output_array[..., j, :] = target
    if need_to_add_last:
        output_array[..., -1, :] = input_array[..., -Q:]
    return output_array


def prepare_half_overlapping(input_array, Q):
    if input_array.shape[-1] < Q:
        return prepare_half_overlapping_small_case(input_array, Q)
    else:
        return prepare_half_overlapping_large_case(input_array, Q)


class RecursiveGammatoneNP(object):
    def __init__(self, N=1, sigma=0.1, xi=0., name=""):
        # if sigma and xi are not arrays, make them arrays
        if type(sigma) != np.ndarray:
            sigma = np.array([sigma])
        if type(xi) != np.ndarray:
            xi = np.array([xi])
        # create filter parameters
        self.N = N
        self.sigma = sigma
        self.low_pass = np.any(xi == 0.)
        self.xi = xi
        self.final_renorm = np.ones(xi.shape)
        self.add_factor = 1. - np.exp(-self.sigma)
        self.name = name

    def forward(self, input_seq, take_modulus=False, hidden_past=None,
                return_last_hidden=False, inplace=False):
        """
        Compute input_seq \ast psi_{\lambda} (possibly followed by a modulus)
        hidden_past is the previous hidden value used for computations

        Returns the new convolved sequence, plus possibly the last hidden
        value, which can be used later on.

        hidden_past is a dictionary with the following keys:
        - 'phase': the last phase phi such that the previous input
           was multiplied by x(t) e^{-i phi}, and then y(t) e^{i phi}
           after the filtering
           By linearity, the subsequent phases should be computed with
           phi + xi*t
           This key is not used if self.low_pass is True
        - k, for k = 1, ..., N: each of these entries is a torch tensor
           of adequate size (e.g. the batch_size + another axis for real
           and imaginary parts)
        - 'grad': the last entry of the hidden value AFTER remodulation

        The 'inplace' option is only relevant for low-pass filters.

        Assume that the first dimension of input_seq is the time. Otherwise,
        input_seq may have an arbitrary shape. Due to the jit compilation, it
        is better to use the same shape multiple times.
        """
        # preparation: creation of the hidden state if not existing
        if hidden_past is None:
            hidden_past = self.create_hidden_past(input_seq.shape)
        if return_last_hidden:
            last_hidden = {}
        # Initial demodulation
        if self.low_pass:
            z = self.demodulate_phi(input_seq, inplace=inplace)
        else:
            res = self.demodulate_psi(input_seq, hidden_past,
                                      return_last_hidden=return_last_hidden)
            z, expo_c = res[0], res[1]
            if return_last_hidden:
                last_hidden['phase'] = res[2]
        # N filters
        for k in range(1, self.N + 1):
            # Filter MA(1) with the adequate initialization
            filter_add_inplace_np(z, self.add_factor, h0=hidden_past[k])
            # Keep a trace of the last hidden value, if required
            if return_last_hidden:
                # size batch (low-pass) in all cases
                last_hidden[k] = z[-1].copy()
        # If required, remodulate and take a last grad1D
        if not(self.low_pass):
            z *= expo_c
            if return_last_hidden:
                last_hidden['grad'] = z[-1].copy()
            # Make a grad1D
            filter_grad1D_inplace_np(z, h0=hidden_past['grad'])
        # Perform the final renormalization
        z *= self.final_renorm
        # squeeze one dimension if the shape is only 1, so that it is
        # completely transparent
        if self.add_factor.size == 1:
            z = z.reshape(z.shape[:-1])
        # Take the modulus if required
        if take_modulus:
            z = np.abs(z, out=z)  # performed INPLACE!
        if return_last_hidden:
            return z, last_hidden
        else:
            return z

    def create_hidden_past(self, input_shape):
        h = {}
        mytype = 'float' if self.low_pass else 'complex'
        newdim = self.add_factor.shape
        for k in range(1, self.N + 1):
            h[k] = np.zeros(input_shape[1:] + newdim, dtype=mytype)
        h['grad'] = np.zeros(input_shape[1:] + newdim, dtype=mytype)
        h['phase'] = np.zeros(input_shape[1:] + (1,), dtype='float')
        return h

    def demodulate_phi(self, input_seq, inplace=False):
        z = input_seq if inplace else input_seq.copy()
        # add one last dimension for the parallelization dimension
        # (useful if multiple phi)
        z = z.reshape(z.shape + (1,))
        if self.add_factor.size > 1:
            z = z * np.ones(self.add_factor.size)
            # NB: this tiles z in memory!
        return z

    def demodulate_psi(self, input_seq, hidden_past,
                       return_last_hidden=False):
        num_other_dims = len(input_seq.shape) - 1
        T = input_seq.shape[0]
        t_range = np.arange(T)
        # Compute the newphase
        newsize = (T,) + create_one_tuples(num_other_dims + 1)
        arg = t_range.reshape(newsize) * self.xi
        # take the past phase accumulator (for continuity)
        phi = hidden_past['phase']  # batch_size x 1
        arg = arg + phi  # broadcast operation: size T x batch_size x wavelets
        # get the complex exponential
        expo_c = np.cos(arg) + 1j * np.sin(arg)
        # first assignment
        z = input_seq.reshape(input_seq.shape + (1,))
        # multiplication with broadcast
        z = z * np.conj(expo_c)
        # z now has shape time x ... x 1
        # we now expand z along the wavelet dimension (last)
        z = z * np.ones(self.add_factor.size)

        # if necessary, record the last phase
        if return_last_hidden:
            next_phase = arg[-1] + self.xi
            return z, expo_c, next_phase
        else:
            return z, expo_c

    def normalize(self, length=int(1e5), normalization='l1'):
        x = np.zeros([length])
        x[0] = 1.
        # compute the impulse response
        impulse_rep = self.forward(x, take_modulus=True)
        # assert that the last value is sufficiently small
        eps = impulse_rep[-1] / np.max(impulse_rep, axis=0)
        assert np.max(eps) < 1e-7
        if normalization == 'l1':
            lp_norm = np.sum(impulse_rep, axis=0)
        elif normalization == 'l2':
            lp_norm = np.sqrt(np.sum(impulse_rep**2, axis=0))
        else:
            raise ValueError('Unknown normalization ' + str(normalization))
        assert len(lp_norm.shape) == 1  # otherwise, problem!
        assert np.max(np.abs(np.imag(lp_norm))) < 1e-7
        self.final_renorm = 1. / np.real(lp_norm)


class HaarNP(object):
    def __init__(self, final_renorm=1.):
        self.final_renorm = 0.5 * np.ones(1)

    def forward(self, input_seq, take_modulus=False,
                hidden_past=None, return_last_hidden=False):
        """
        Computes input_seq \ast h (possibly with a modulus)
        hidden_past is the previous hidden value used for computations

        Returns the new convolved sequence, plus possibly the
        last hidden value, which can be used back

        input_seq has shape T x whatever

        hidden_past is an array of size whatever; it should contain
        the previous value of input_seq (like 'grad') for recursive
        gammatone

        NB: it does NOT act inplace!
        """
        z = input_seq.copy()
        if return_last_hidden:
            last_hidden = z[-1].copy()
        if hidden_past is None:
            hidden_past = self.create_hidden_past(input_seq)
        # Make a grad1D
        filter_grad1D_inplace_np(z, h0=hidden_past)
        # Perform the final renormalization
        z *= self.final_renorm
        # Take the modulus if required
        if take_modulus:
            z = np.abs(z, out=z)
        if return_last_hidden:
            return z, last_hidden
        else:
            return z

    def create_hidden_past(self, input_seq):
        return np.zeros(input_seq.shape[1:], dtype=input_seq.dtype)

    def normalize(self, *args, **kwargs):
        # Nothing to do, because by construction it is already normalized
        # in l^1 norm!
        pass


class RecurrentScatteringNP(object):
    def __init__(self, J, Q, N, only_J=False, include_haar=False, joint=False,
                 len_fft_joint=6, max_length=int(1e5), normalization='l1',
                 avg_U1_before_joint=True):
        # build the wavelets
        phi, psi1, psi2 = build_wavelet_family(J, Q, N, RecursiveGammatoneNP,
                                               HaarNP, verbose=False,
                                               include_haar=include_haar,
                                               max_length=max_length,
                                               normalization=normalization)
        if only_J:
            self.phi = {J: phi[J]}
        else:
            self.phi = phi
        self.psi2 = psi2
        # create the subwavelet families
        psi1m, meta1 = create_psi1m(J, psi1, RecursiveGammatoneNP,
                                    only_J=only_J, include_haar=include_haar)
        self.meta1 = meta1
        self.psi1m = psi1m
        # Precompute meta2:
        self.meta2 = create_meta2(sorted(list(psi1m.keys())), meta1,
                                  list(self.psi2.keys()))
        self.is_cuda = False
        self.only_J = only_J
        self.include_haar = include_haar
        self.joint = joint
        self.avg_U1_before_joint = avg_U1_before_joint
        if self.joint:
            self.len_fft = len_fft_joint
            self.hann_window = np.hanning(self.len_fft + 2)[1:-1]
            # we truncate the hann window so that the first and last points
            # are not strictly ignored (even if assigned with small weights
            assert self.only_J  # joint only works in only_J mode
            # just another sanity check
            for j2 in self.meta2.keys():
                assert len(self.meta2[j2]) == 1  # otherwise

    def first_order(self, xj0, h_all):
        xj1, hj1 = self._first_order_normal(xj0, h_all)
        if self.joint:
            # perform the pre-averaging in an inplace fashion
            if self.avg_U1_before_joint:
                for j in self.phi.keys():
                    xj1[j], hj1[j]['low_for_joint'] = self.phi[j].forward(
                        xj1[j], inplace=True,
                        hidden_past=h_all[1][j]['low_for_joint'],
                        return_last_hidden=True)
            # now, work along frequencies
            for j in self.phi.keys():
                xj1[j] = self._localized_fourier_along_freqs(
                    xj1[j], is_real=True)
        return xj1, hj1

    def _first_order_normal(self, xj0, h_all):
        xj1 = {}
        hj1 = {}
        for j in self.psi1m.keys():
            if j in xj0.keys():
                hj1[j] = {}
                if self.include_haar:
                    x_haar, hj1[j]['haar'] = self.psi1m[0].forward(
                        xj0[j].reshape(xj0[j].shape + (1,)), take_modulus=True,
                        hidden_past=h_all[1][j]['haar'],
                        return_last_hidden=True)
                temp, hj1[j]['other'] = self.psi1m[j].forward(
                    xj0[j], take_modulus=True, return_last_hidden=True,
                    hidden_past=h_all[1][j]['other'])
                if self.include_haar:
                    xj1[j] = np.concatenate([x_haar, temp], axis=-1)
                else:
                    xj1[j] = temp
        # manage the case of order 1
        if not(self.only_J) and self.include_haar:
            hj1[1] = {}
            xj1[1], hj1[1]['haar'] = self.psi1m[0].forward(
                xj0[1].reshape(xj0[1].shape + (1,)), take_modulus=True,
                hidden_past=h_all[1][1]['haar'], return_last_hidden=True)
        return xj1, hj1

    def second_order(self, xj1, h_all):
        xj2, hj2 = self._second_order_normal(xj1, h_all,
                                             take_modulus=not(self.joint))
        if self.joint:
            for j2 in xj2.keys():
                xj2[j2] = self._localized_fourier_along_freqs(xj2[j2])
        return xj2, hj2

    def _second_order_normal(self, xj1, h_all, take_modulus=True):
        xj2 = {}
        hj2 = {}
        for j2 in self.psi2.keys():
            tocat = []
            if j2 in self.meta2.keys():
                # for each j, we gather all the possible arrays
                # NB: here, there might be a bug if the order is not
                # respected!
                for (j1, ind_start, num_take) in self.meta2[j2]:
                    tocat.append(
                        np.take(xj1[j1], np.arange(num_take), axis=-1))
                # concatenate them
                x_to_filter = np.concatenate(tocat, axis=-1)
                # filter it
                xj2[j2], hj2[j2] = self.psi2[j2].forward(
                    x_to_filter, take_modulus=take_modulus,
                    hidden_past=h_all[2][j2], return_last_hidden=True)
        return xj2, hj2

    def _localized_fourier_along_freqs(self, z, is_real=False):
        # z: np array of size (T, B, C), complex!
        # prepare for the filtering along j1
        z_half_overlap = prepare_half_overlapping(z, self.len_fft)
        z_half_overlap *= self.hann_window
        # filter along the last dimension
        z_joint = np.fft.fft(z_half_overlap, axis=-1)
        # remove half of the coordinates if real
        if is_real:
            num_to_keep = int(np.ceil(z_joint.shape[-1] // 2)) + 1
            z_joint = z_joint[..., :num_to_keep]
        # take the modulus (inplace)
        z_joint = np.abs(z_joint, out=z_joint)
        # reshape
        newshape = z.shape[:-1]
        newshape = newshape + (z_joint.shape[-2] * z_joint.shape[-1],)
        z_joint = z_joint.reshape(newshape)
        return z_joint

    def averaging(self, xj0, xj1, xj2, h_all):
        S = {}
        h_low = {}
        for j in self.phi.keys():
            # gather the corresponding indices
            # 0 and 1st order: nothing to do!
            tocat = [xj0[j].reshape(xj0[j].shape + (1,)), xj1[j]]
            for j2 in sorted(xj2.keys()):
                if j2 < j:
                    if not(self.joint):
                        for (a, b, c) in self.meta2[j2]:
                            # NB: possible bug here due to the order in meta2
                            if a == j:
                                tocat.append(
                                    np.take(xj2[j2], b + np.arange(c),
                                            axis=-1))
                    else:
                        # by construction, meta2[j2] has only 1 element, so:
                        tocat.append(xj2[j2])
            if len(tocat) > 0:
                U = np.concatenate(tocat, axis=-1)
                S[j], h_low[j] = self.phi[j].forward(
                    U, inplace=True, hidden_past=h_all['low'][j],
                    return_last_hidden=True)
        return S, h_low

    def forward(self, x, hidden_past=None, return_last_hidden=False):
        # check if x is an array (to duplicate), or a dictionary
        if type(x) == dict:
            xj0 = {k: np.asarray(x[k], dtype='complex') for k in x.keys()}
        else:
            xj0 = {j: np.asarray(x.copy(), dtype='complex')
                   for j in self.phi.keys()}
        if hidden_past is None:
            h_all = self.create_hidden_past()
        else:
            h_all = hidden_past
        # ORDER 1
        xj1, hj1 = self.first_order(xj0, h_all)
        # ORDER 2
        xj2, hj2 = self.second_order(xj1, h_all)
        # Final averaging
        S, h_low = self.averaging(xj0, xj1, xj2, h_all)
        # Take the real part:
        S = {k: np.real(S[k]) for k in S.keys()}
        if return_last_hidden:
            return S, {1: hj1, 2: hj2, 'low': h_low}
        else:
            return S

    def create_hidden_past(self):
        # we just have to create a dictionary of None, with the good keys
        # by recursion, they will be transmitted to the functions,
        # which will manage the shapes
        h1 = {}
        for j in self.psi1m.keys():
            if j != 0:
                h1[j] = {'other': None}
                if self.include_haar:
                    h1[j]['haar'] = None
                if self.joint:
                    h1[j]['low_for_joint'] = None
        if self.include_haar:
            h1[1] = {'haar': None}
        h2 = {}
        for j2 in self.psi2.keys():
            if j2 in self.meta2.keys():
                h2[j2] = None
        h_low = {}
        for j in self.phi.keys():
            h_low[j] = None
        return {1: h1, 2: h2, 'low': h_low}
