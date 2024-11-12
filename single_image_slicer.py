import pickle
import os
import matplotlib.pyplot as plt

def get_slices(index, nii_images, image_names):

    selected_image = nii_images[index]
    selected_image_name = image_names[index]

    # Ottieni i dati dell'immagine
    image_data = selected_image.get_fdata()
    print(f"Nome dell'immagine selezionata: {selected_image_name}")
    print(f"Forma dell'immagine: {image_data.shape}")

    # Seleziona tre slice a diverse profondità:

    # Slice verticali:
    x_center = image_data.shape[0]//2

    sagittal_slice = image_data[x_center, :,:]  #nell'asse Z ci va un intervallo di valori dato dal file json. 

    slices = [
        (sagittal_slice, f"{selected_image_name}_slice_center"),
    ]
    return slices

def plot_slice(slice):
    plt.imshow(slice, cmap="gray")
    plt.show()

def save_slices(slices, image_name):
    slices_folder = "slices_output"
    os.makedirs(slices_folder, exist_ok=True)
    
    subfolder_name = image_name.split('_')[0]  # estrae "sub-verseXXX" da "sub-verseXXX_seg-vert_msk.nii"
    subfolder_path = os.path.join(slices_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Scorre tutte le slice e salva ciascuna nell'ordine specificato
    for slice_data, slice_name in slices:
        # Costruisce il percorso completo del file
        slices_path = os.path.join(subfolder_path, f"{slice_name}.png")

        # Salva la slice come immagine PNG
        plt.imsave(slices_path, slice_data, cmap="gray")
        print(f"Salvato {slice_name} a {slices_path}")
