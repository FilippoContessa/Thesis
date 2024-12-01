import torch
from vae_model import VariationalAutoEncoder
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Specifica il dispositivo
device = "cpu"
# Inizializza il modello
vae = VariationalAutoEncoder(input_dim=200 * 200, h_dim=400, z_dim=20)
vae = vae.to(device)

# Carica i pesi
vae.load_state_dict(torch.load("vae_model.pth", map_location=device,weights_only=True))

# Metti il modello in modalità valutazione
vae.eval()


# Trasformazioni delle immagini (devono essere identiche a quelle usate durante il training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Carica l'immagine
img_path = "augmented_slices_opt/sub-verse004/vertebra_16/flipping.png"  # Sostituisci con il percorso della tua immagine
img = Image.open(img_path).convert("L")  # Scala di grigi
img_tensor = transform(img).view(-1, 200 * 200).to(device)

# Inferenza
with torch.no_grad():
    reconstructed, _, _ = vae(img_tensor)

# Converti i tensor in immagini per visualizzarle
original = img_tensor.view(200, 200).cpu().numpy()
reconstruction = reconstructed.view(200, 200).cpu().numpy()

# Visualizza l'immagine originale e quella ricostruita
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(original, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(reconstruction, cmap="gray")

plt.show()
