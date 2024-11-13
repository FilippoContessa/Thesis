import pickle
import os
import matplotlib.pyplot as plt
from access_to_json_files import get_vertebrae_coordinates

def get_slices(index, nii_images, image_names, json_files):
    selected_image = nii_images[index]
    selected_image_name = image_names[index]
    json_info = json_files[index] # file .json corrispondente all'immagine selezionata

    # Ottieni i dati dell'immagine
    image_data = selected_image.get_fdata()
    print(f"Nome dell'immagine selezionata: {selected_image_name}")
    print(f"Forma dell'immagine: {image_data.shape}")

    margin = 5
    coordinates = get_vertebrae_coordinates(index, json_files, 1)

    x_center = image_data.shape[0]//2

    sagittal_slice = image_data[x_center, :,coordinates[2]- margin: coordinates[2] + margin ]  #nell'asse Z ci va un intervallo di valori dato dal file json. 
    
    #TODO: fissato un margine, vediamo di ottenere la posizione della x,y,z centrale di ciascuna vertebra, e poi ricavarne un quadratino di dimensioni fissate attorno
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
