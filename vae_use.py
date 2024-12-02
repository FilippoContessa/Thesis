import torch
from vae_model import VariationalAutoEncoder
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

INPUT_DIM = 100 * 100
DEVICE = "cpu"
# Specifica il dispositivo

# Inizializza il modello
vae = VariationalAutoEncoder(input_dim=INPUT_DIM, h_dim=400, z_dim=20)
vae = vae.to(DEVICE)

# Carica i pesi
vae.load_state_dict(torch.load("vae_model.pth", map_location=DEVICE,weights_only=True))

# Metti il modello in modalità valutazione
vae.eval()


# Trasformazioni delle immagini (devono essere identiche a quelle usate durante il training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Carica l'immagine
img_path = "augmented_slices/sub-verse004/vertebra_20/original_angle0_scale1.png" 
img = Image.open(img_path).convert("1")  
img_tensor = transform(img).view(-1, INPUT_DIM).to(DEVICE)

# Inferenza
with torch.no_grad(): # Disabilita il calcolo del gradiente
    reconstructed, _, _ = vae(img_tensor)

# Converti i tensor in immagini per visualizzarle
original = img_tensor.view(100, 100).cpu().numpy()
reconstruction = reconstructed.view(100, 100).cpu().numpy()

# Visualizza l'immagine originale e quella ricostruita
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(original, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(reconstruction, cmap="gray")
plt.show()