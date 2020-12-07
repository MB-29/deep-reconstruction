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
n_epoch = 5
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

batch = next(iter(train_loader))

# print('model')
model = VariationalAutoEncoder(input_dim, embedding_dim)
print('optimizer')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train
training_losses = []
print(f'Training loop')

for epoch_index in range(n_epoch):
    print(f'Epoch {epoch_index}')
    epoch_loss = 0

    for batch_index, (image_input, label) in enumerate(train_loader):
        if label != 5:
            continue
        if batch_index > 1000:
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

    training_losses.append(epoch_loss)


def generate_image():
    embedding_sample = model.sample(torch.zeros(embedding_dim),
                          torch.ones(embedding_dim))
    generated_sample = model.decode(embedding_sample)
    return generated_sample.data.numpy().reshape((image_size, image_size, n_channels))

def vector_to_image(vector):
    return vector.reshape((image_size, image_size, n_channels))


T = 10000
x = batch[0].view(batch_size, input_dim)

sampler = Sampler(input_dim, embedding_dim, model, x, T)
sampled_output = vector_to_image(sampler.sample_gibbs().data)

plt.imshow(sampled_output, cmap='gray')
plt.show()
