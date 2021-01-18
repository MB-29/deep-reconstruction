from math import log
import numpy as np
import torch
from scipy.stats import multivariate_normal
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from vae import VariationalAutoEncoder


class Sampler:

    def __init__(self, autoencoder, x0, missing_data_indices, T):

        self.autoencoder = autoencoder

        self.x0 = x0
        self.missing_data_indices = missing_data_indices
        self.T = T
        self.samples = [x0.data.numpy().copy()]

    @torch.no_grad()
    def sample_pseudo_gibbs(self):
        x = self.x0.clone()

        for t in range(self.T):
            z = self.autoencoder.embed(x)
            generated_x = self.autoencoder.decode(z)
            x[self.missing_data_indices] = generated_x[self.missing_data_indices]
            self.samples.append(x.data.numpy().copy())

        return self.samples

    @torch.no_grad()
    def sample_metropolis(self):
        x = self.x0.clone()
        z = self.autoencoder.embed(x)
        d = self.autoencoder.embedding_dim
        standard_normal = MultivariateNormal(torch.zeros(d), torch.eye(d))
        for t in range(self.T):
            # proposal = self.autoencoder.embed(x)
            mean, log_var = self.autoencoder.encode(x)
            proposal = self.autoencoder.sample(mean, log_var)
            z_log_likelihood = standard_normal.log_prob(z)
            proposal_log_likelihood = standard_normal.log_prob(proposal)
            decoded_proposal = self.autoencoder.decode(proposal)
            decoded_z = self.autoencoder.decode(z)
            phi_z = F.binary_cross_entropy(
                decoded_z, x, reduction='mean')
            phi_proposal = F.binary_cross_entropy(
                decoded_proposal, x, reduction='mean')
            multi_normal = MultivariateNormal(
                mean, log_var.exp() * torch.eye(d))
            psi_z = multi_normal.log_prob(z)
            psi_proposal = multi_normal.log_prob(proposal)

            log_rho = proposal_log_likelihood + phi_proposal + psi_z - \
                (z_log_likelihood + phi_z + psi_proposal)

            u = np.random.uniform()
            if u <= torch.exp(log_rho):
                z = proposal
                x[self.missing_data_indices] = decoded_proposal[self.missing_data_indices]
            self.samples.append(x.data.numpy().copy())

        return self.samples
