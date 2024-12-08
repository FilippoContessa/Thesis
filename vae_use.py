import torch
from vae_model import ConvVariationalAutoEncoder
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Impostazioni
INPUT_DIM = 96 * 96 
DEVICE = "cpu"

# Inizializza il modello
vae = ConvVariationalAutoEncoder(20)
vae = vae.to(DEVICE)

# Carica i pesi
vae.load_state_dict(torch.load("conv_vae_model.pth", map_location=DEVICE,weights_only=True))

# Metti il modello in modalità valutazione
vae.eval()

# Trasformazioni delle immagini (devono essere identiche a quelle usate durante il training)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converte in Tensor 
])

# Carica l'immagine
img_path = "augmented_slices/sub-verse004/vertebra_20/original_angle0_scale1.png"
img = Image.open(img_path).convert("L")  # Usa 'L' per immagine in scala di grigi
img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Aggiungi una dimensione batch (1, 1, 192, 192)

# Inferenza
with torch.no_grad():  # Disabilita il calcolo del gradiente
    reconstructed, _, _ = vae(img_tensor)

# Converti i tensor in immagini per visualizzarle
original = img_tensor.squeeze(0).cpu().numpy()  # Rimuovi la dimensione batch (1)
reconstruction = reconstructed.squeeze(0).cpu().numpy()  # Rimuovi la dimensione batch (1)

# Visualizza l'immagine originale e quella ricostruita
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(original[0], cmap="gray")  # Aggiungi [0] per selezionare il canale unico

plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(reconstruction[0], cmap="gray")  # Aggiungi [0] per selezionare il canale unico
plt.axis('off')
plt.show()