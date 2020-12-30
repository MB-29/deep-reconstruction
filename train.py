import torch
import torch.optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from vae import VariationalAutoEncoder
from sampler import Sampler

data_path = '/Users/matthieu/Projets/Data/'
image_size = 28
n_channels = 1

batch_size = 1
n_epochs = 20
input_dim = image_size * image_size * n_channels
embedding_dim = 64
learning_rate = 1e-3

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_path, train=True, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_path, train=False, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)


def train(model, optimizer, n_epochs):
    training_loss_values = []
    print(f'Training loop')

    for epoch_index in range(n_epochs):
        print(f'Epoch {epoch_index}')
        epoch_loss = 0

        for batch_index, (image_input, label) in enumerate(train_loader):
            if batch_index > 5000:
                break
            if batch_index % 100 == 0:
                print(f'Batch index {batch_index}')

            input_vector = image_input.view(batch_size, input_dim)
            generated_sample, embedded_mean, embedded_log_var = model(input_vector)

            loss = model.loss(input_vector, generated_sample,
                            embedded_mean, embedded_log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss

        training_loss_values.append(epoch_loss)
    return training_loss_values

def generate_image(model):
    embedding_sample = model.sample(torch.zeros(embedding_dim),
                          torch.ones(embedding_dim))
    generated_sample = model.decode(embedding_sample)
    return generated_sample.data.numpy().reshape((image_size, image_size, n_channels))

def vector_to_image(vector):
    return vector.reshape((image_size, image_size, n_channels))

def plot_generated_samples(model, n_samples=5):
    for sample_index in range(1, n_samples+1):
        ax = plt.subplot(1, n_samples, sample_index)
        ax.imshow(generate_image(model), cmap='gray')
        plt.axis('off')


def plot_images(samples):
    for sample_index, sample in enumerate(samples):
        ax = plt.subplot(1, len(samples), sample_index+1)
        ax.imshow(sample, cmap='gray')
        plt.axis('off')


batch = next(iter(train_loader))

# TRAIN
model = VariationalAutoEncoder(input_dim, embedding_dim)
print('optimizer')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(n_epochs)
training_loss_values = train(model, optimizer, n_epochs)

# RECONSTRUCTION

def reconstruct(model, x0, missing_data_indices, T, method='pseudo_Gibbs'):
    sampler = Sampler(model, x0.clone(), missing_data_indices, T)
    sampling_method = sampler.sample_pseudo_gibbs if method=='pseudo_Gibbs' else sampler.sample_metropolis
    sampled_vectors = sampling_method()
    sampled_images = np.array([vector_to_image(vector)
                               for vector in sampled_vectors])
    error = (x0 - torch.Tensor(sampled_vectors)).norm(dim=1)
    return sampled_images, error


T = 100
t_sample_values = np.linspace(0, T-1, num=10, dtype=int)
missing_data_indices = np.arange(200, 500)

x = batch[0][0].view(input_dim)
x0 = x.data.clone()

# add noise
x.data[missing_data_indices] = torch.abs(torch.randn(size=(len(missing_data_indices),)))

# output
sampled_images, error = reconstruct(model, x0, missing_data_indices, T)
