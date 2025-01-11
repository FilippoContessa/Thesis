import torch
from vae_model import ConvVariationalAutoEncoder
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# Parametri Generali
INPUT_DIM = 96 * 96 
Z_DIM = 30
DEVICE = "cpu"
MODEL_PATH = "Models/conv_vae_model_Beta=8e-05.pth"
img_path = "augmented_slices2.0/sub-verse005/vertebra_21/original_angle0_scale1.png"
output_path="Generated_Imgs/Beta=8e-05"
os.makedirs(output_path, exist_ok=True)


# Inizializza il modello
vae = ConvVariationalAutoEncoder(z_dim=Z_DIM)
vae = vae.to(DEVICE)
vae.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,weights_only=True))

vae.eval()  # Imposta il modello in modalità di valutazione

# Trasformazioni delle immagini 
transform = transforms.Compose([
    transforms.ToTensor(),  # Converte in Tensor 
])

"""
# --- Ricostruzione di un'immagine ---
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
#plt.savefig('Reconstruction_B=5e-6.png')
plt.show()
"""

with torch.no_grad():  # Disabilita il calcolo del gradiente
    # --- Generazione di immagini dallo spazio latente ---
    for index in range(100):

            random_latent_vector = torch.randn(1, Z_DIM).to(DEVICE)  # Campiona dallo spazio latente
            generated_img = vae.decode(random_latent_vector)  # Decodifica l'immagine

            # Processa il tensor dell'immagine generata per la visualizzazione
            generated_img = generated_img.squeeze(0).squeeze(0).cpu().numpy()  # Rimuovi dimensione batch e canale

            # Visualizza l'immagine generata
            plt.imshow(generated_img, cmap="gray")
            plt.title(f"Generated Image {index+1}")
            plt.axis("off")
            save_path=os.path.join(output_path,f"Generated_Image_{index+1}.png")
            plt.savefig(save_path) 
            print(f"creata e salvata immagine numero: {index+1}")
