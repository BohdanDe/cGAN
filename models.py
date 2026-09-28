import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=128) # was 128

        self.realism_checker = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(in_features=7*7*64, out_features=256), # was 256
            nn.ReLU(),
        )

        self.if_number_is_a_number_checker = nn.Sequential(
            nn.Linear(in_features=128+256, out_features=64), # was 64
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, image, label):
        realism_embed_dim = self.realism_checker(image)
        l = self.embedding(label)
        return self.if_number_is_a_number_checker(torch.cat([realism_embed_dim, l], dim=1))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.noise = nn.Sequential(
            nn.Linear(in_features=100, out_features=64), # wa 64
            nn.ReLU(),
        )

        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=128) # was 128

        self.model = nn.Sequential(
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=1, padding=0),  # (96, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),   # (48, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),   # (24, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(24, 1, kernel_size=4, stride=2, padding=3),    # (1, 28, 28)
            nn.Tanh()
        )

    def forward(self, noise, label):
        n = self.noise(noise)
        l = self.embedding(label)
        latent_dim = torch.cat([n, l], dim=1)
        latent_dim = latent_dim.view(latent_dim.size(0), 128+64, 1, 1)
        return self.model(latent_dim)