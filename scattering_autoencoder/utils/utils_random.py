import numpy as np
from joblib import Parallel, delayed
import numba


def uniform_law(omega_min, omega_max):
    return (omega_max - omega_min) * np.random.rand(1) + omega_min


def log_uniform_law(omega_min, omega_max, min_val=0.01 * np.pi):
    om = max(min_val, omega_min)
    logsampled = uniform_law(0, np.log2(omega_max) - np.log2(om))
    return om * np.power(2., logsampled)


def log_distance(y, others):
    return np.min(np.abs(np.log2(y / others)))


def uni_distance(y, others):
    return np.min(np.abs(y - others))


@numba.jit
def sample_omegas_simple(num=1, delta=1.0, omega_min=0., omega_max=np.pi,
                         max_niter=10, ensure_comp_in_interval=None,
                         log_uniform=True):
    """
    Sample omegas between omega_min and omega_max uniformly,
    but make sure that the minimal distance between them is at least of delta
    """
    if log_uniform:
        sampling_fun = log_uniform_law
        distance_fun = log_distance
    else:
        sampling_fun = uniform_law
        distance_fun = uni_distance

    omegas = np.zeros(num)
    if ensure_comp_in_interval is None:
            omegas[0] = sampling_fun(omega_min, omega_max)
    else:
        om = ensure_comp_in_interval[0]  # omega_minus, omega_plus
        op = ensure_comp_in_interval[1]  # omega_minus, omega_plus
        omegas[0] = sampling_fun(om, op)
    for i in range(1, num):
        omegas[i] = sampling_fun(omega_min, omega_max)
        iter = 0
        while (iter < max_niter) and \
                (distance_fun(omegas[i], omegas[:i]) < delta):
            omegas[i] = sampling_fun(omega_min, omega_max)
            iter += 1
    return omegas


def sample_omegas(num_components=1, delta=1.0, omega_min=0., omega_max=np.pi,
                  num_examples=16, n_jobs=1, ensure_comp_in_interval=None,
                  log_sample=True):
    # TODO : fix the parallel generation
    # The parallel generation is not working so we only do it with 1 subprocess
    if n_jobs > 1:
        n_jobs = 1
    if n_jobs == 1:
        # Not carried out in parallel in this case
        omegas = [sample_omegas_simple(
            num=num_components, delta=delta, omega_min=omega_min,
            omega_max=omega_max,
            ensure_comp_in_interval=ensure_comp_in_interval,
            log_uniform=log_sample)
            for _ in range(num_examples)]
    else:
        omegas = Parallel(n_jobs=n_jobs)(
            delayed(sample_omegas_simple)(
                num=num_components, delta=delta, omega_min=omega_min,
                omega_max=omega_max,
                ensure_comp_in_interval=ensure_comp_in_interval)
            for _ in range(num_examples))
    omegas = np.array(omegas)
    return omegas


def create_overlap_add(N, num_components, num_examples,
                       normalize=False, split_for_keras=True,
                       random_translation=False, num_iter=3,
                       log_sample=False, omega_min=0.1,
                       omega_max=0.5 * np.pi,
                       ensure_abs_lower_than_1=True):
    # define the signals
    if random_translation:
        factor = num_iter + 1
    else:
        factor = num_iter
    n = np.arange(0, N)
    omegas = sample_omegas(
        num_components=num_components, num_examples=factor * num_examples,
        log_sample=log_sample, omega_min=omega_min, omega_max=omega_max)
    phis = 2 * np.pi * np.random.rand(factor * num_examples, num_components)
    a = np.random.exponential(1, size=(factor * num_examples, num_components))
    if normalize:
        a /= np.sum(a, axis=1).reshape(-1, 1)
    x = np.sum(np.cos(np.einsum('ij, k -> ijk', omegas, n) +
                      np.expand_dims(phis, axis=-1)) *
               np.expand_dims(a, axis=-1), axis=1)

    # Multiply the signals by hanning:
    g = np.hanning(N)
    xg = x * g.reshape(1, -1)

    # Make the overlap-add
    overlap = np.zeros([num_examples, int(0.5 * (factor + 1) * N)])
    for k in range(factor):
        overlap[:, k * int(N / 2):N + k * int(N / 2)] += xg[k::factor, :]
    if ensure_abs_lower_than_1:
        overlap /= np.max(np.abs(overlap), axis=-1).reshape(-1, 1)
    # Split between past and future if necessary
    if split_for_keras:
        if random_translation:
            # create the array of random translations
            trans = np.random.randint(0, high=int(N / 2), size=(num_examples,))
            J = np.add.outer(trans, np.arange(int(N / 2), int(1.5 * N),
                                              dtype=int))
            last = (trans + int(1.5 * N)).reshape(-1, 1)
            I = np.arange(num_examples).reshape(-1, 1)
            return overlap[I, J], overlap[I, last]
        else:
            return overlap[:, int(N / 2):int(1.5 * N)],\
                overlap[:, int(1.5 * N)].reshape(-1, 1)
    else:
        return overlap, omegas


def create_mixture_cosine(N, num_components, num_examples, normalize=False,
                          split_for_keras=False, law='exponential'):
    n = np.arange(0, N + 1)
    omegas = sample_omegas(num_components=num_components,
                           num_examples=num_examples, log_sample=False,
                           omega_min=0.1, omega_max=0.9 * np.pi)
    phis = 2 * np.pi * np.random.rand(num_examples, num_components)
    if law == 'exponential':
        a = np.random.exponential(1, size=(num_examples, num_components))
    elif law == 'equal':
        a = np.ones([num_examples, num_components]) / float(num_components)
    if normalize:
        a /= np.sum(a, axis=1).reshape(-1, 1)
    x = np.sum(np.cos(np.einsum('ij, k -> ijk', omegas, n) +
               np.expand_dims(phis, axis=-1)) * np.expand_dims(a, axis=-1),
               axis=1)
    if split_for_keras:
        return x[:, :-1], x[:, -1].reshape(-1, 1)
    else:
        return x, omegas


def create_overlap_add_without_boundaries(size_block, num_comp, batch_size,
                                          normalize=True, num_iter=5,
                                          log_sample=True, omega_min=2e-2,
                                          omega_max=0.5 * np.pi):
    x, _ = create_overlap_add(size_block, num_comp, batch_size,
                              normalize=normalize, split_for_keras=False,
                              num_iter=num_iter, log_sample=True,
                              omega_min=omega_min, omega_max=omega_max)
    x = x[:, int(size_block / 2):-int(size_block / 2)].T
    return x


def add_gaussian_noise(x, sigma=1.0):
    z = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
    return x + z


def multiplicative_noise(x, sigma=1.0):
    e = - sigma * np.log(np.random.uniform(low=0.0, high=1.0, size=x.shape))
    return e * x


def create_super_random_signal(seq_len, batch_size=256, num_comp=2,
                               omega_min=1e-3, omega_max=np.pi, j_block_min=5,
                               j_block_max=10):
    # assert that batch_size can be divided by 2
    j_size_block = np.random.randint(j_block_min, high=j_block_max + 1,
                                     size=(2,))
    size_block = np.asarray(np.power(2, j_size_block), dtype=int)
    num_block = np.asarray(np.ceil(2 * seq_len / size_block) + 1, dtype=int)
    x1 = create_overlap_add_without_boundaries(
        size_block[0], num_comp, int(batch_size / 2), log_sample=True,
        omega_min=omega_min, omega_max=omega_max, num_iter=num_block[0])
    x2 = create_overlap_add_without_boundaries(
        size_block[1], num_comp, int(batch_size / 2), log_sample=True,
        omega_min=omega_min, omega_max=omega_max, num_iter=num_block[1])
    x = np.concatenate([x1[:seq_len], x2[:seq_len]], axis=1)
    x = add_gaussian_noise(x)
    x = multiplicative_noise(x)
    return x


def add_heteroschedastic_noise(x, sigma_min=1e-2, sigma_max=0.7):
    logsigma = np.random.uniform(
        low=np.log(sigma_min), high=np.log(sigma_max), size=(x.shape[-1],))
    sigma = np.exp(logsigma)
    z = np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    return x + sigma * z


def create_harmonic_overlap_add(N, num_harmonics, num_examples,
                                num_iter=3, log_sample=True, omega_min=0.15,
                                omega_max=0.5 * np.pi,
                                ensure_abs_lower_than_1=True):
    factor = num_iter
    n = np.arange(0, N)
    # sample base frequency
    omegas = sample_omegas(
        num_components=1, num_examples=factor * num_examples,
        log_sample=log_sample, omega_min=omega_min,
        omega_max=omega_max / num_harmonics)
    # add harmonics
    omegas = np.squeeze(omegas)
    omegas = np.outer(omegas, np.arange(num_harmonics) + 1)
    # amplitudes and phase
    phis = 2 * np.pi * np.random.rand(factor * num_examples, num_harmonics)
    a = np.ones([factor * num_examples, num_harmonics])
    a /= np.sum(a, axis=1).reshape(-1, 1)
    # compute the signals
    x = np.sum(np.cos(np.einsum('ij, k -> ijk', omegas, n) +
                      np.expand_dims(phis, axis=-1)) *
               np.expand_dims(a, axis=-1), axis=1)

    # Multiply the signals by hanning:
    g = np.hanning(N)
    xg = x * g.reshape(1, -1)

    # Make the overlap-add
    overlap = np.zeros([num_examples, int(0.5 * (factor + 1) * N)])
    for k in range(factor):
        overlap[:, k * int(N / 2):N + k * int(N / 2)] += xg[k::factor, :]
    if ensure_abs_lower_than_1:
        overlap /= np.max(np.abs(overlap), axis=-1).reshape(-1, 1)
    # Split between past and future if necessary
    return overlap, omegas


def create_harmonic_overlap_add_without_boundaries(size_block, num_comp,
                                                   batch_size, num_iter=5,
                                                   log_sample=True,
                                                   omega_min=0.15,
                                                   omega_max=0.5 * np.pi):
    x, _ = create_harmonic_overlap_add(size_block, num_comp, batch_size,
                                       num_iter=num_iter, log_sample=True,
                                       omega_min=omega_min,
                                       omega_max=omega_max)
    x = x[:, int(size_block / 2):-int(size_block / 2)].T
    return x
