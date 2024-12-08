import os
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from vae_model import ConvVariationalAutoEncoder, ImageDataset, loss_function  

INPUT_DIM = 96 * 96 
BATCH_SIZE = 128
EPOCHS = 5
DEVICE = "cpu"

# Trasformazioni per le immagini
transform = transforms.Compose([
    transforms.ToTensor(),  # Converte in Tensor
])

# Carica il dataset
dataset = ImageDataset("augmented_slices", transform=transform)
print(f"Il numero di immagini nel db è: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# VAE Model
vae = ConvVariationalAutoEncoder(z_dim=20)
vae = vae.to(DEVICE)

# Ottimizzatore
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training Loop
vae.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for imgs in dataloader:
        imgs = imgs.to(DEVICE)  # Immagini non devono essere "flattened" per le CNN
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
torch.save(vae.state_dict(), f"conv_vae_model.pth")
