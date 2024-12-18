import torch
from vae_model import ConvVariationalAutoEncoder
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

INPUT_DIM = 96 * 96 
Z_DIM = 30
DEVICE = "cpu"
img_path = "augmented_slices/sub-verse005/vertebra_21/original_angle0_scale1.png"

# Inizializza il modello
vae = ConvVariationalAutoEncoder(z_dim=Z_DIM)
vae = vae.to(DEVICE)
vae.load_state_dict(torch.load("conv_vae_model.pth", map_location=DEVICE,weights_only=True))
vae.eval() # Imposta il modello in modalità di valutazione

# Trasformazioni delle immagini 
transform = transforms.Compose([
    transforms.ToTensor(),  # Converte in Tensor 
])

img = Image.open(img_path).convert("L")  
img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Aggiungi una dimensione batch (1, 1, 192, 192)

with torch.no_grad():  # Disabilita il calcolo del gradiente
    reconstructed, _, _ = vae(img_tensor)

# Converti i tensor in immagini per visualizzarle
original = img_tensor.squeeze(0).cpu().numpy()  # Rimuovi la dimensione batch (1)
reconstruction = reconstructed.squeeze(0).cpu().numpy()  # Rimuovi la dimensione batch (1)

# Visualizza
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(original[0], cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(reconstruction[0], cmap="gray")
plt.axis('off')
plt.show()