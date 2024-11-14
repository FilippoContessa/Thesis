import pickle
import os
import matplotlib.pyplot as plt
from access_to_json_files import get_vertebrae_coordinates
import numpy as np

def get_dynamic_x_center(image_data):
    # Calcola l'intensità media su ogni slice lungo l'asse X (profondità)
    intensity_means = np.mean(image_data, axis=(1, 2))

    # Trova l'indice con l'intensità massima, che dovrebbe corrispondere alla colonna vertebrale
    x_center = np.argmax(intensity_means)

    # Se desideri una posizione più stabile, puoi anche considerare la mediana dei primi valori massimi
    return x_center

def get_slices(index, nii_images, image_names, json_files):
    selected_image = nii_images[index]
    selected_image_name = image_names[index]
    json_info = json_files[index] # file .json corrispondente all'immagine selezionata

    # Ottieni i dati dell'immagine
    image_data = selected_image.get_fdata()
    print(f"Nome dell'immagine selezionata: {selected_image_name}")
    print(f"Forma dell'immagine: {image_data.shape}")
    
    #TODO: implementa un modo per calcolare il margine variabile
    margin = 14
    
    x_center = get_dynamic_x_center(image_data)
    slices = []

    for vertebrae_index in range(1,len(json_info)):
        coordinates = get_vertebrae_coordinates(index, json_files, vertebrae_index)

        sagittal_slice = image_data[x_center,:,:]
        single_vertebrae_slice = image_data[x_center, 
        int(round(coordinates[1])) - margin : int(round(coordinates[1])) + margin, 
        :]

        #nell'asse Y ci va un intervallo di valori dato dal file json, L'asse z regola la larghezza dell'immagine.
    
        slices.append((single_vertebrae_slice, f"{selected_image_name}_vertebra_{vertebrae_index}_slice"))
        slices.append((sagittal_slice, f"Sagittal_{selected_image_name}"))

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