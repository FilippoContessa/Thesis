import os
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from vae_model import VariationalAutoEncoder

# Dataset personalizzato per le tue immagini
class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.file_paths = [os.path.join(root, file)
                           for root, _, files in os.walk(folder)
                           for file in files if file in {"flipping.png", "rotation.png", "scaling.png"}]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        img = Image.open(img_path).convert("L")  # Scala di grigi
        if self.transform:
            img = self.transform(img)
        return img

# Trasformazioni per le immagini
transform = transforms.Compose([
    transforms.ToTensor(),  # Converte in Tensor e normalizza in [0, 1]
    transforms.Normalize((0.5,), (0.5,))  # Normalizza i valori (media=0.5, std=0.5)
])

# Carica il dataset
dataset = ImageDataset("augmented_slices_opt", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# VAE Model
vae = VariationalAutoEncoder(input_dim=200 * 200, h_dim=400, z_dim=20)  # Cambia input_dim alle tue dimensioni
vae = vae.to("cuda" if torch.cuda.is_available() else "cpu")

# Loss Function e Ottimizzatore
def loss_function(x_reconstructed, x, mu, sigma):
    # Ricostruzione Loss (es. MSE)
    reconstruction_loss = nn.MSELoss()(x_reconstructed, x)

    # KL Divergence
    kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

    return reconstruction_loss + kl_divergence

optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training Loop
epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
vae.train()

for epoch in range(epochs):
    total_loss = 0
    for imgs in dataloader:
        imgs = imgs.view(imgs.size(0), -1).to(device)  # Flatten delle immagini
        optimizer.zero_grad()

        # Forward Pass
        x_reconstructed, mu, sigma = vae(imgs)

        # Loss
        loss = loss_function(x_reconstructed, imgs, mu, sigma)
        loss.backward()

        # Aggiorna i pesi
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

# Salva il modello
torch.save(vae.state_dict(), "vae_model.pth")

# Carica e valuta il modello
vae.eval()
with torch.no_grad():
    for imgs in dataloader:
        imgs = imgs.view(imgs.size(0), -1).to(device)
        x_reconstructed, mu, sigma = vae(imgs)

        # Visualizza alcune immagini originali e ricostruite
        original = imgs[0].view(200, 200).cpu().numpy()
        reconstructed = x_reconstructed[0].view(200, 200).cpu().numpy()

        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(original, cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title("Reconstructed")
        plt.imshow(reconstructed, cmap="gray")
        plt.show()
        break  # Mostra solo un batch
