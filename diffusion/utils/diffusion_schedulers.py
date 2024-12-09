"""Schedulers for Denoising Diffusion Probabilistic Models"""

import math

import numpy as np
import torch
import torch.nn.functional as F


class CategoricalDiffusion(object):
    def __init__(self, T, schedule):
        # Diffusion steps
        self.T = T

        # Noise schedule
        if schedule == 'linear':
            b0 = 1e-4
            bT = 2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
                0)  # Generate an extra alpha for bT
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

        beta = self.beta.reshape((-1, 1, 1))
        eye = np.eye(2).reshape((1, 2, 2))
        ones = np.ones((2, 2)).reshape((1, 2, 2))

        self.Qs = (1 - beta) * eye + (beta / 2) * ones

        Q_bar = [np.eye(2)]
        for Q in self.Qs:
            Q_bar.append(Q_bar[-1] @ Q)
        self.Q_bar = np.stack(Q_bar, axis=0)

    def __cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

    def sample(self, x0_onehot, t):
        # Select noise scales
        # x0_onehot: N, 1, 1, 2
        # self.Q_bar: 1001, 2, 2
        Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)

        xt = torch.matmul(x0_onehot, Q_bar.reshape((Q_bar.shape[0], 1, 2, 2)))
        return torch.bernoulli(xt[..., 1].clamp(0, 1))

    def consistency_sample(self, x0_onehot, t, t2):
        """
        Args:
            x0_onehot: B, N, N, 2
            t: B
            t2: B, next time step of t, t2 < t
        Returns:
            xt: B, N, N
            xt2: B, N, N
        """
        Q_bar_t2 = torch.from_numpy(self.Q_bar[t2]).float().to(x0_onehot.device)
        Q_bar_t = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)

        xt2 = torch.matmul(x0_onehot, Q_bar_t2.reshape((Q_bar_t2.shape[0], 1, 2, 2)))
        xt2 = torch.bernoulli(xt2[..., 1].clamp(0, 1))  # B, N, N

        xt = torch.matmul(F.one_hot(xt2.long(), num_classes=2).float(), (torch.linalg.inv(Q_bar_t2) @ Q_bar_t).reshape((Q_bar_t.shape[0], 1, 2, 2)))
        xt = torch.bernoulli(xt[..., 1].clamp(0, 1))

        return xt, xt2

class InferenceSchedule(object):
    def __init__(self, inference_schedule="linear", T=1000, inference_T=1000):
        self.inference_schedule = inference_schedule
        self.T = T
        self.inference_T = inference_T

    def __call__(self, i):
        assert 0 <= i < self.inference_T

        if self.inference_schedule == "linear":
            t1 = self.T - int((float(i) / self.inference_T) * self.T)
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        elif self.inference_schedule == "cosine":
            t1 = self.T - int(
                np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int(
                np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        else:
            raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))
