import torch
from torch import nn
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class ConvVariationalAutoEncoder(nn.Module):
    def __init__(self, z_dim):
        super(ConvVariationalAutoEncoder, self).__init__()
        
        # Encoder : # input channel (scala di grigi = 1) , output channel = n° feature map ottenute, kernel size = dimensione del filtro(4x4) , stride = passo con cui scorre il filtro, padding = aggiunta pixel bordi.
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),  # 1x96x96 -> 32x48x48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x48x48 -> 64x24x24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x24x24 -> 128x12x12
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 128x12x12 -> 256x6x6
            nn.ReLU(),
            nn.Flatten()  # 256x6x6 -> 9216
        )
        self.fc_mu = nn.Linear(256 * 6 * 6, z_dim)
        self.fc_sigma = nn.Linear(256 * 6 * 6, z_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(z_dim, 256 * 6 * 6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 256x6x6 -> 128x12x12
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x12x12 -> 64x24x24
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x24x24 -> 32x48x48
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 32x48x48 -> 1x96x96
            nn.Sigmoid()  # Sigmoid per normalizzare nell'intervallo [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        sigma = self.fc_sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 256, 6, 6)  # Reshape to match decoder input:  256 n° feature map, 12x12 è la dimensione voluta.
        x_reconstructed = self.decoder(h)
        return x_reconstructed
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, sigma


class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.file_paths = [os.path.join(root, file)
                           for root, _, files in os.walk(folder)
                           for file in files if file.startswith("flipped") or file.startswith("original")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        img = Image.open(img_path).convert("L")  # Grayscale images
        if self.transform:
            img = self.transform(img)
        return img


def loss_function(x_reconstructed, x, mu, sigma, beta):
    reconstruction_loss = nn.BCELoss()(x_reconstructed, x)

    # KL Divergence
    kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

    return reconstruction_loss + beta * kl_divergence
