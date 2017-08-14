import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import math


class GaussianMLP(nn.Module):
    """
    Gaussian Multilayer Perceptron Policy.

    Policy that samples actions from a gaussian distribution with a
    learned (but state invariant) standard deviation.
    """

    def __init__(self, obs_dim, action_dim,  hidden_dims=(32, 32),
                 init_std=1.0, nonlin=F.tanh, optimizer=optim.Adam):

        super(GaussianMLP, self).__init__()

        self.hidden_layers = nn.ModuleList()

        self.hidden_layers += [nn.Linear(obs_dim, hidden_dims[0])]

        for l in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[l]
            out_dim = hidden_dims[l + 1]
            self.hidden_layers += [nn.Linear(in_dim, out_dim)]

        self.out = nn.Linear(hidden_dims[-1], action_dim)

        self.log_stds = nn.Parameter(
            torch.ones(1, action_dim) * np.log(init_std)
        )

        self.nonlin = nonlin

    def forward(self, x):
        for l in self.hidden_layers:
            x = self.nonlin(l(x))

        means = self.out(x)

        log_stds = self.log_stds.expand_as(means)

        stds = torch.exp(log_stds)

        return means, log_stds, stds

    def get_action(self, means, stds):
        action = torch.normal(means, stds)
        return action.detach()

    def log_likelihood(self, x, means, log_stds, stds):
        var = stds.pow(2)

        log_density = -(x - means).pow(2) / (
            2 * var) - 0.5 * math.log(2 * math.pi) - log_stds

        return log_density.sum(1)

    def kl_divergence(self, mean0, log_std0, std0,
                            mean1, log_std1, std1):
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1)


    # TODO: add entropy
    # maybe encapsulate distribution related functions