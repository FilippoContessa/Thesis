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
    x_quarter = image_data.shape[0]//4
    x_center = image_data.shape[0]//2
    x_three_quarters = (image_data.shape[0] * 3) // 4

    sagittal_slice_quarter = image_data[x_quarter, :, :]
    sagittal_slice = image_data[x_center, :,:]  
    sagittal_slice_three_quarters = image_data[x_three_quarters, :, :]
#TODO: AGGIUNGI INFO SUL NOME DELLE SLICE
    slices = [sagittal_slice_quarter, sagittal_slice, sagittal_slice_three_quarters]
    return slices

def plot_slice(slice):
    plt.imshow(slice, cmap="gray")
    plt.show()

import os
import matplotlib.pyplot as plt

def save_slices(slices, image_names):
    slices_folder = "slices_output"
    os.makedirs(slices_folder, exist_ok=True)
    
    # Scorre tutte le slice e salva ciascuna
    for i in range(len(slices)):
        slice = slices[i]
        slice_name = f"{image_names[i]}_slice_{i}.png"  # Definisce il nome della slice FIXME: IN REALTà E SBAGLIATO, MODIFICALO QUANDO PERFEZIONERAI GET SLICES
        slices_path = os.path.join(slices_folder, slice_name)  # Costruisce il percorso completo del file

        # Salva la slice come immagine PNG
        plt.imsave(slices_path, slice, cmap="gray")
        print(f"Salvato {slice_name} a {slices_path}")

