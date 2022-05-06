from torch import nn
import numpy as np

class Discriminator(nn.Module):

    def __init__(self, img_size=[1, 28, 28]):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(np.prod(img_size, dtype=np.int32), 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        # image.shape = [batch_size, 1, 28, 28]
        image = image.reshape(image.shape[0], -1)

        prob = self.main(image)
        return prob