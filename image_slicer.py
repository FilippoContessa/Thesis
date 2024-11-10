import pickle
import os
import matplotlib.pyplot as plt

# Carica le immagini e i nomi dal file .pkl
with open("images_data.pkl", "rb") as f:
    nii_images, image_names = pickle.load(f)


# Scorre tra tutte le immagini e salva le slice come file separati

# FIXME: Questo ciclo è troppo lungo e complesso. Dovresti spezzarlo in più funzioni. Per farlo funzionare correttamente devi spezzare il ciclo for in tanti piccoli range.
index = 12

selected_image = nii_images[index]
selected_image_name = image_names[index]

# Ottieni i dati dell'immagine
image_data = selected_image.get_fdata()
print(f"Nome dell'immagine selezionata: {selected_image_name}")
print(f"Forma dell'immagine: {image_data.shape}")

# Seleziona e salva tre slice a diverse profondità:

# Slice verticali:
x_quarter = image_data.shape[0]//4
x_center = image_data.shape[0]//2
x_three_quarters = (image_data.shape[0] * 3) // 4

sagittal_slice_quarter = image_data[x_quarter, :, :]
sagittal_slice = image_data[x_center, :,:]  
sagittal_slice_three_quarters = image_data[x_three_quarters, :, :]



# Visualizza le slice
plt.imshow(sagittal_slice, cmap="gray")
plt.title(f"Slice 1 di {selected_image_name}")
plt.show()