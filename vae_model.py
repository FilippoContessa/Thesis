import torch
from torch import nn
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

# Da input img -> hidden layer -> mu e sigma, -> repar trick -> decoder -> Output img

class VariationalAutoEncoder(nn.Module): 
    def __init__(self, input_dim, h_dim, z_dim): #input_dim per MNISt è 28*28=784, h_dim è il numero di neuroni dello strato nascosto, z_dim è la dimensione dello spazio latente
        super().__init__() # call the constructor of the parent class
        
        # encoder 
        self.img_2hid = nn.Linear(input_dim, h_dim) # Mandi l'input all'hidden layer
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x): #q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z): #p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma*epsilon
        x_reconstructed = self.decode(z_new)
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
        img = Image.open(img_path).convert("1")  
        if self.transform:
            img = self.transform(img)
        return img

def loss_function(x_reconstructed, x, mu, sigma):
    # Ricostruzione Loss (es. MSE)
    reconstruction_loss = nn.MSELoss()(x_reconstructed, x)

    # KL Divergence
    kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

    return reconstruction_loss + kl_divergence