import os

import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F


class VariationalAutoEncoder(nn.Module):

    def __init__(self, input_dim, embedding_dim, inter_dim=100):
        super(VariationalAutoEncoder, self).__init__()
        self.fc_in_1 = nn.Linear(in_features=input_dim, out_features=inter_dim)
        self.fc_in_2 = nn.Linear(
            in_features=inter_dim, out_features=embedding_dim)
        self.fc_in_3 = nn.Linear(
            in_features=inter_dim, out_features=embedding_dim)
        self.fc_out_1 = nn.Linear(
            in_features=embedding_dim, out_features=inter_dim)
        self.fc_out_2 = nn.Linear(
            in_features=inter_dim, out_features=input_dim)

    def encode(self, x):
        x = F.relu(self.fc_in_1(x))
        return self.fc_in_2(x), self.fc_in_3(x)

    def decode(self, x):
        x = F.relu(self.fc_out_1(x))
        return torch.sigmoid(self.fc_out_2(x))

    def sample(self, mean, log_var):
        std = torch.exp(log_var/2)
        normal = torch.rand_like(std)
        return mean + std * normal

    def forward(self, x):
        mean, log_var = self.encode(x)
        embedding_sample = self.sample(mean, log_var)
        generated_sample = self.decode(embedding_sample)
        return generated_sample, mean, log_var

    def embed(self, x):
        mean, log_var = self.encode(x)
        return self.sample(mean, log_var)

    def loss(self, x, generated_sample, mean, log_var):
        reconstruction_loss = F.binary_cross_entropy(
            generated_sample, x, reduction='sum')
        kl_divergence = -0.5 * \
            torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reconstruction_loss + kl_divergence
