import pickle
import os
import matplotlib.pyplot as plt
#TODO: dovrei importare image_slicer e in questo file concentrarmi SOLO nell'estendere lo slicer a tutte le immagini del db.
# Carica le immagini e i nomi dal file .pkl
with open("images_data.pkl", "rb") as f:
    nii_images, image_names = pickle.load(f)

# Percorso per salvare le slice
output_folder = "slices_output"
os.makedirs(output_folder, exist_ok=True)
print(len(nii_images))
# Scorre tra tutte le immagini e salva le slice come file separati
# FIXME: Questo ciclo è troppo lungo e complesso. Dovresti spezzarlo in più funzioni. Per farlo funzionare correttamente devi spezzare il ciclo for in tanti piccoli range.
for i in range(len(nii_images)):
    selected_image = nii_images[i]
    selected_image_name = image_names[i]

    # Ottieni i dati dell'immagine
    image_data = selected_image.get_fdata()
    print(f"Nome dell'immagine selezionata: {selected_image_name}")
    print(f"Forma dell'immagine: {image_data.shape}")

    # Seleziona e salva tre slice a diverse profondità
    casual_slice_1 = image_data[:, :, image_data.shape[2] // 4]
    casual_slice_2 = image_data[:, :, image_data.shape[2] // 3]
    casual_slice_3 = image_data[:, :, image_data.shape[2] // 2]

    # Salva le slice come file .pkl individuali
    for idx, slice_data in enumerate([casual_slice_1, casual_slice_2, casual_slice_3], start=1):
        slice_name = f"{selected_image_name}_slice{idx}.pkl"
        slice_path = os.path.join(output_folder, slice_name)
        with open(slice_path, "wb") as slice_file:
            pickle.dump(slice_data, slice_file)
        print(f"Salvata {slice_name} a {slice_path}")
