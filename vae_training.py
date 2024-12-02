import os
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from vae_model import VariationalAutoEncoder, ImageDataset, loss_function

INPUT_DIM = 100 * 100
BATCH_SIZE = 128
EPOCHS = 15
DEVICE = "cpu"

# Trasformazioni per le immagini
transform = transforms.Compose([
    transforms.ToTensor(),  # Converte in Tensor e normalizza in [0, 1]
    transforms.Normalize((0.5,), (0.5,))  # Normalizza i valori (media=0.5, std=0.5)
])

# Carica il dataset
dataset = ImageDataset("augmented_slices", transform=transform)
print(f"il numero di immagini caricate è: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# VAE Model
vae = VariationalAutoEncoder(input_dim=INPUT_DIM, h_dim=400, z_dim=20)
vae = vae.to(DEVICE)

# Ottimizzatore
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training Loop
vae.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for imgs in dataloader:
        imgs = imgs.view(imgs.size(0), -1).to(DEVICE)  # Flatten delle immagini
        optimizer.zero_grad()

        # Forward Pass
        x_reconstructed, mu, sigma = vae(imgs)

        # Loss
        loss = loss_function(x_reconstructed, imgs, mu, sigma)
        loss.backward()

        # Aggiorna i pesi
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {round(total_loss / len(dataloader),3)}")

# Salva il modello
torch.save(vae.state_dict(), f"vae_model.pth")