import subprocess
from datetime import datetime
import numpy as np


def get_git_revision_hash():
    label = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return label.decode('utf-8')


def get_git_revision_short_hash():
    label = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return label.decode('utf-8')


def dec_gen(callable):
    """
    A decorator to create a generator from a function
    """
    def as_generator(*args, **kwargs):
        while True:
            yield callable(*args, **kwargs)
    return as_generator


def get_timestamp():
    return "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())


def log_subsample(z, num_points=int(1e3)):
    assert len(z.shape) == 1  # otherwise not valid
    T = z.size
    t_range = np.logspace(start=0, stop=np.log10(T), dtype=int, base=10,
                          num=num_points)
    t_range = np.unique(t_range)
    t_range = t_range[t_range < T]
    t_range = t_range[t_range >= 0]
    return t_range, z[t_range]


def dictionary_to_array(h):
    # compute the size of the array
    any_key = list(h.keys())[0]
    out = np.zeros((len(h.keys()),) + h[any_key].shape, dtype=h[any_key].dtype)
    for i, k in enumerate(sorted(list(h.keys()))):
        out[i] = h[k]
    return out
