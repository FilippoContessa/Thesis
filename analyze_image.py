import pickle
import matplotlib.pyplot as plt

# Carica le immagini e i nomi dal file .pkl
with open("images_data.pkl", "rb") as f:
    nii_images, image_names = pickle.load(f)

# Seleziona e visualizza un'immagine specifica

indice = 0  
selected_image = nii_images[indice]
selected_image_name = image_names[indice]

# Ottieni i dati dell'immagine
image_data = selected_image.get_fdata()
print(f"Nome dell'immagine selezionata: {selected_image_name}")
print(f"Forma dell'immagine: {image_data.shape}")

# Visualizza una fetta dell'immagine
casual_slice = image_data[image_data.shape[0] // 4,:, :]
plt.imshow(casual_slice.T, cmap="gray", origin="lower")
plt.title(f"Fetta casuale di {selected_image_name}")
plt.show()
