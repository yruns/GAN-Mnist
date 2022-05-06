import numpy as np
from torch import nn

class Generator(nn.Module):

    def __init__(self, latent_dim, img_size= [1, 28, 28]):
        super(Generator, self).__init__()
        self.img_size = img_size

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, np.prod(img_size)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x.shape = (batch_size, latent_dim)
        image = self.main(x)

        return image.reshape(x.shape[0], *self.img_size)
