import numpy as np


# Functions to derive the sigma parameter from a desired bandwidth
def get_sigma_discrete_phi(N, r, B):
    root_r_sqinv = np.power(r, -2. / float(N))
    sigma = np.arccosh((root_r_sqinv - np.cos(0.5 * B)) / (root_r_sqinv - 1.))
    return sigma


def get_sigma_discrete_psi(N, r, xi, B):
    root_r = np.power(r, 1. / N)
    C_sq = 2 * (1 - np.cos(xi))
    K1 = 2 * np.sin(xi) / (N * C_sq)
    K2 = (2 / (N * C_sq)) * (np.cos(xi) - (1. - 1. / N) *
                             (np.sin(xi)**2) / C_sq)
    p = np.zeros(3)
    p[0] = (B**2) * np.power(root_r, 4.)
    p[1] = 2 * (root_r**2) * (2 * ((root_r**2) - 1) - K2 * (B**2))
    p[2] = (B * K2)**2 - (K1**2 + 4 * ((root_r**2) - 1) * K2)
    g_sigma = np.max(np.roots(p))
    sigma = 2 * np.arcsinh(0.5 / np.sqrt(g_sigma))
    return sigma


def compute_size_scattering(J, Q, include_haar=True, joint=False,
                            len_fft_joint=6, return_detail=False):
    """
    Instead of using a formula, we detail a script
    """
    # order 0
    order0 = 1

    # order 1
    order1 = (J - 1) * Q
    if include_haar:
        order1 += 1
    if joint:
        order1 = compute_num_joint_channels(order1, len_fft_joint,
                                            is_real=True)
    # order 2
    if joint:
        order2 = 0
        for j2 in range(2, J):
            num_sub_channels = (j2 - 1) * Q
            order2 += compute_num_joint_channels(num_sub_channels,
                                                 len_fft_joint)
    else:
        order2 = Q * int((J - 2) * (J - 1) / 2)
        if include_haar:
            order2 += J - 1
    if return_detail:
        return order0, order1, order2
    else:
        return order0 + order1 + order2


def compute_num_joint_channels(num_previous_channels, len_fft, is_real=False):
    if is_real:
        num_fft = int(np.ceil(len_fft / 2)) + 1
    else:
        num_fft = len_fft
    if num_previous_channels < len_fft:
        return num_fft
    else:
        half = len_fft // 2
        k = num_previous_channels // half
        if num_previous_channels % half == 0:
            return (k - 1) * num_fft
        else:
            return k * num_fft


def build_parameters_wavelet_family(J, Q, N, r=np.sqrt(0.5)):
    # Low-pass
    factor = max(2 / (1 + np.power(2, 6 / Q)), 0.67)
    B_low = factor * np.pi * np.power(0.5, np.arange(J, dtype=float))
    sigma_low = get_sigma_discrete_phi(N, r, B_low)
    # Wavelets for second-order
    xi2 = np.zeros(J)
    sigma2 = np.zeros(J)
    xi2[-1] = np.min(B_low)  # for J
    sigma2[-1] = get_sigma_discrete_psi(N, r, xi2[-1], np.min(B_low))
    for j in range(J - 2, -1, -1):
        xi2[j] = xi2[j + 1] * 2
        sigma2[j] = sigma2[j + 1] * 2
    # Wavelets for first-order
    lambda1 = (np.arange(Q, J * Q + 1, dtype=float) / float(Q))
    xi1 = np.min(xi2) * np.power(2, lambda1[::-1] - np.min(lambda1))
    root_Q = np.power(0.5, 1. / float(Q))
    B1 = 2 * (1 - root_Q) * xi1 / (1 + root_Q)
    sigma1 = np.zeros(B1.size)
    for i in range(sigma1.size):
        sigma1[i] = get_sigma_discrete_psi(N, r, xi1[i], B1[i])
    return sigma_low, sigma1, xi1, lambda1, sigma2, xi2


def create_one_tuples(n):
    t = ()
    for _ in range(n):
        t += (1,)
    return t


def create_meta2(sorted_lambda1, meta1, psi2_keys):
    meta2 = {}
    for j2 in psi2_keys:
        origin = []
        ind_start = 0
        for j1 in sorted_lambda1:
            if j2 < j1:
                coords = [ind_k for ind_k in range(len(meta1[j1]))
                          if meta1[j1][ind_k] < j2]
                if len(coords) > 0:
                    # coords should be of the shape [0, ..., n - 1]
                    # for some n, so:
                    n = max(coords) + 1
                    # j1, ind where it will start, number of coordinates
                    origin.append((j1, ind_start, n))
                    ind_start += n
        if len(origin) > 0:
            meta2[j2] = origin
    return meta2


def split_parallel_gammatones(psi, keys, gammatone_obj):
    # assume that there is a one-to-one correspondance between psi and keys
    # (for instance, assume that the keys are sorted!)
    psi_split = {}
    for ind_k in range(len(keys)):
        k = keys[ind_k]
        psi_split[k] = gammatone_obj(
            N=psi.N, sigma=np.array([psi.sigma[ind_k]]),
            xi=np.array([psi.xi[ind_k]]))
        psi_split[k].final_renorm[0] = psi.final_renorm[ind_k]
    return psi_split


def merge_parallel_gammatones(psi, gammatone_obj):
    # assume that psi is a dictionary of gammatone_obj's
    ordered_keys = sorted(list(psi.keys()))
    sigma = np.zeros(len(ordered_keys))
    xi = np.zeros(len(ordered_keys))
    for ind_k in range(len(ordered_keys)):
        sigma[ind_k] = psi[ordered_keys[ind_k]].sigma
        xi[ind_k] = psi[ordered_keys[ind_k]].xi[0]
    psi_all = gammatone_obj(N=psi[ordered_keys[0]].N, sigma=sigma, xi=xi)
    # share the normalizing factor
    for ind_k in range(len(ordered_keys)):
        psi_all.final_renorm[ind_k] = psi[ordered_keys[ind_k]].final_renorm[0]
    return psi_all, ordered_keys


def build_psi(N, sigma, xi, gammatone_obj, haar_obj, include_haar=True,
              max_length=int(1e5), normalization='l1'):
    psi = {}
    if include_haar:
        psi[0.] = haar_obj()
    psi['other'] = gammatone_obj(N=N, sigma=sigma, xi=xi)
    for k in psi.keys():
        psi[k].normalize(length=max_length, normalization=normalization)
    return psi


def build_wavelet_family(J, Q, N, gammatone_obj, haar_obj, r=np.sqrt(0.5),
                         max_length=int(1e5), verbose=False,
                         include_haar=True, normalization='l1'):
    # I) Parameters
    sigma_low, sigma1, xi1, lambda1, sigma2, xi2 = \
        build_parameters_wavelet_family(J, Q, N, r=r)
    # II) Actual family
    # Low pass: in one block:
    phi = gammatone_obj(N=N, sigma=sigma_low, xi=np.zeros(sigma_low.size))
    phi.normalize(length=max_length, normalization=normalization)
    if verbose:
        print('phi built!')
    # First order
    psi1 = build_psi(N, sigma1, xi1, gammatone_obj, haar_obj,
                     include_haar=include_haar, max_length=max_length,
                     normalization=normalization)
    if verbose:
        print('psi1 built!')
    # Second order
    psi2 = build_psi(N, sigma2, xi2, gammatone_obj, haar_obj,
                     include_haar=include_haar, max_length=max_length,
                     normalization=normalization)
    if verbose:
        print('psi2 built')
    # Resplit them
    all_j = np.arange(1, J + 1)
    phi = split_parallel_gammatones(phi, all_j, gammatone_obj)
    if include_haar:
        psi1_split = {0.: psi1[0.]}
    else:
        psi1_split = {}
    psi1_split.update(split_parallel_gammatones(psi1['other'], lambda1,
                                                gammatone_obj))
    if include_haar:
        psi2_split = {0.: psi2[0.]}
    else:
        psi2_split = {}
    psi2_split.update(split_parallel_gammatones(psi2['other'], all_j,
                                                gammatone_obj))

    return phi, psi1_split, psi2_split


def create_psi1m(J, psi1, gammatone_obj, only_J=False, include_haar=True):
    # 0 is a particular case = Haar.
    # It needs to be added to ALL j's afterwards,
    meta1 = {}
    if include_haar:
        psi1m = {0.: psi1.pop(0.)}
        meta1[1] = [0.]  # will be added afterwards
    else:
        psi1m = {}
    # but it is only computed once.
    start = J if only_J else 1
    for j in range(start, J + 1):
        subdic = {k: psi1[k] for k in psi1.keys() if k < j}
        if len(subdic) > 0:
            psi1m[j], meta1[j] = merge_parallel_gammatones(
                subdic, gammatone_obj)
            if include_haar:
                meta1[j] = [0.] + meta1[j]  # will be added afterwards
    return psi1m, meta1
