import torch

from torch.autograd import Variable
try:
    from .multiproc_helper import MPGenerator, my_default_collate
except Exception:  # import error
    from multiproc_helper import MPGenerator, my_default_collate


class SyntheticScattering(object):
    def __init__(self, generator, M, alpha, max_gen=600, length_return=9,
                 dim_scattering=10, past_size=4, take_relu=True, **kwargs):
        self.scatpred = generator
        self.cov_sqrt = M
        self.ampl = alpha.view(-1, 1)
        self.max_gen = max_gen
        self.length_return = length_return
        self.dim_scat = dim_scattering
        self.past_size = past_size
        self.relu = torch.nn.ReLU()
        self.take_relu = take_relu

    def get(self):
        with torch.no_grad():
            # First, generate a long sequence of scattering
            S_up = []
            for _ in range(self.past_size):
                # Initialize it with noise with the adequate covariance
                S_up.append(torch.matmul(torch.randn(1, self.dim_scat),
                                        self.cov_sqrt))
            for t in range(self.max_gen):
                # take the last sample and make a deterministic prediction
                S_last_past = torch.cat(
                    [v.unsqueeze(-1) for v in S_up[-self.past_size:]], dim=-1)
                S_deter_pred = self.scatpred.forward(S_last_past)[..., 0]
                # add the noise
                Z = torch.matmul(torch.randn(S_deter_pred.size()), self.cov_sqrt)
                S_pred = S_deter_pred + Z
                S_up.append(S_pred)
            # concatenate, truncate and rescale
            S_up = torch.cat([v.unsqueeze(-1) for v in S_up], dim=-1)
            S_up = S_up[..., -self.length_return:]
            S_up[:, 1:] = S_up[:, 1:] * self.ampl
            # rectify
            if hasattr(self, 'take_relu'):
                if self.take_relu:
                    S_rect = self.relu(S_up)
                else:
                    S_rect = S_up
            else:
                S_rect = self.relu(S_up)
        # return the result without the batch dimension
        return S_rect[0]


def create_generator_SynthScat(*args, batch_size=128, num_workers=0,
                               queue_size=10, **kwargs):
    synthscat = SyntheticScattering(*args, **kwargs)
    mpg = MPGenerator(synthscat, my_default_collate, num_workers=num_workers,
                      queue_size=queue_size, batch_size=batch_size)
    return mpg
