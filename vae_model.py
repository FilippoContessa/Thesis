import torch
from torch import nn


# Da input img -> hidden layer -> mu e sigma, -> par trick -> decoder -> Output img
class VariationalAutoEncoder(nn.Module): 
    def __init__(self, input_dim, h_dim=200, z_dim=20): #input_dim per MNISt è 28*28=784, h_dim è il numero di neuroni dello strato nascosto, z_dim è la dimensione dello spazio latente
        super().__init__() # call the constructor of the parent class
        
        # encoder 
        self.img_2hid = nn.Linear(input_dim, h_dim) ##Mandi l'input all'hidden layer
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


if __name__ == "__main__":
    x = torch.randn(4, 28*28)
    vae = VariationalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)


import os
from torchvision.utils import save_image

def save_images(originals, reconstructions, output_folder, prefix="epoch", limit=10):
    """
    Salva immagini originali e ricostruite in un'unica immagine concatenata.

    Args:
        originals (torch.Tensor): Batch di immagini originali.
        reconstructions (torch.Tensor): Batch di immagini ricostruite.
        output_folder (str): Cartella di output.
        prefix (str): Prefisso per il nome del file.
        limit (int): Numero massimo di immagini da salvare.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Seleziona le prime `limit` immagini
    originals = originals[:limit]
    reconstructions = reconstructions[:limit]

    # Normalizza le immagini ricostruite
    originals = originals * 0.5 + 0.5  # Inverti la normalizzazione
    reconstructions = reconstructions * 0.5 + 0.5

    # Concatenazione: una riga per ogni immagine
    concatenated = torch.cat((originals, reconstructions), dim=3)  # Concatenazione orizzontale

    # Salva immagine
    save_image(concatenated, os.path.join(output_folder, f"{prefix}_reconstructions.png"))
