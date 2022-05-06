import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import nn
from torchvision.utils import save_image
from torch.nn import functional as F
from model.generator import Generator
from model.discriminator import Discriminator


def train():
    batch_size = 64
    latent_dim = 96
    img_size = [1, 28, 28]
    num_epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    dataset = MNIST("mnist_data", train=True, download=True,
                                         transform=transforms.Compose(
                                             [
                                                 transforms.Resize(28),
                                                 transforms.ToTensor(),
                                                 #  torchvision.transforms.Normalize([0.5], [0.5]),
                                             ]
                                         )
                                         )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Define the model
    generator = Generator(latent_dim, img_size).to(device)
    discriminator = Discriminator(img_size).to(device)

    # Define the optimizer
    g_optimizer = Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
    d_optimizer = Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)

    # Define the loss
    loss_fn = nn.BCELoss()
    labels_one = torch.ones(batch_size, 1).to(device)
    labels_zero = torch.zeros(batch_size, 1).to(device)

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # Train the discriminator
            d_optimizer.zero_grad()
            # Generate fake images
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)

            # Compute the loss
            d_loss = loss_fn(discriminator(fake_images), labels_zero) + loss_fn(discriminator(images), labels_one)
            d_loss.backward()
            d_optimizer.step()

            # Train the generator
            g_optimizer.zero_grad()
            # Generate fake images
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)

            # Compute the loss
            g_loss = loss_fn(discriminator(fake_images), labels_one)
            g_loss.backward()
            g_optimizer.step()

            if i % 50 == 0:
                print(f"step:{len(dataloader) * epoch + i}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}")

            if i % 1000 == 0:
                image = fake_images[:16].data
                save_image(image, f"images/image_{len(dataloader) * epoch + i}.png", nrow=4)
                print(f"image_{len(dataloader) * epoch + i}.png saved")



if __name__ == '__main__':
    train()

