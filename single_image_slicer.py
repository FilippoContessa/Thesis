import os
import pickle
import matplotlib.pyplot as plt
from access_to_json_files import get_vertebrae_Y_next_distance, get_vertebrae_Y_prev_distances, get_vertebrae_coordinates
import numpy as np
from scipy.ndimage import label, find_objects

def get_dynamic_x_center(image_data):
    # Calcola l'intensità media su ogni slice lungo l'asse X (profondità)
    intensity_means = np.mean(image_data, axis=(1, 2))
    # Trova l'indice con l'intensità massima, che dovrebbe corrispondere alla colonna vertebrale
    x_center = np.argmax(intensity_means)
    return x_center

#FIXME: Se una vertebra è fatta da due componenti che non sono connesse non funziona


def extract_largest_connected_component(slice_data):
    """
    Isola la componente connessa più grande in una maschera binaria.
    """
    labeled, num_features = label(slice_data)  # Etichettatura delle componenti connesse
    if num_features == 0:  # Se non ci sono componenti, restituisci la maschera vuota
        return slice_data
    component_sizes = np.bincount(labeled.ravel())  # Conta i pixel per etichetta
    component_sizes[0] = 0  # Ignora lo sfondo
    largest_label = component_sizes.argmax()  # Trova l'etichetta della componente più grande
    largest_component = labeled == largest_label  # Crea una maschera con solo la componente più grande
    return largest_component

def get_slices(index, nii_images, image_names, json_files):
    selected_image = nii_images[index]
    selected_image_name = image_names[index]
    json_info = json_files[index]  # File .json corrispondente all'immagine selezionata

    # Ottieni i dati dell'immagine
    image_data = selected_image.get_fdata()
    print(f"Nome dell'immagine selezionata: {selected_image_name}")
    print(f"Forma dell'immagine: {image_data.shape}")

    margin_prev = get_vertebrae_Y_prev_distances(index, json_files)
    margin_next = get_vertebrae_Y_next_distance(index, json_files)
    fixed_margin = 13
    bias = 3
    x_center = get_dynamic_x_center(image_data)
    slices = []

    for vertebrae_index in range(1, len(json_info)):
        coordinates = get_vertebrae_coordinates(index, json_files, vertebrae_index)
        sagittal_slice = image_data[x_center, :, :]
        #TODO: Col nuovo algoritmo cosa cambia per la prima e l'ultima?
        
        # Estrazione della regione della vertebra
        if vertebrae_index == 1:
            single_vertebrae_slice = image_data[x_center,
                                                int(round(coordinates[1])) - fixed_margin - bias:
                                                int(round(coordinates[1])) + margin_next[vertebrae_index-2] + bias, :]
        elif vertebrae_index == len(json_info) - 1:
            single_vertebrae_slice = image_data[x_center,
                                                int(round(coordinates[1])) - margin_prev[vertebrae_index-2] - bias:
                                                int(round(coordinates[1])) + fixed_margin + bias, :]
        else:
            single_vertebrae_slice = image_data[x_center,
                                                int(round(coordinates[1])) - margin_prev[vertebrae_index-2] - bias:
                                                int(round(coordinates[1])) + margin_next[vertebrae_index-2] + bias, :]
        
        # Isola la componente connessa più grande
        single_vertebrae_slice = extract_largest_connected_component(single_vertebrae_slice)
        
        # Aggiungi le slice alla lista
        slices.append((single_vertebrae_slice, f"{selected_image_name}_vertebra_{vertebrae_index}_slice"))
        slices.append((sagittal_slice, f"Sagittal_{selected_image_name}"))

    return slices

def plot_slice(slice):
    plt.imshow(slice, cmap="gray")
    plt.show()

def save_slices(slices, image_name):
    slices_folder = "slices_output"
    os.makedirs(slices_folder, exist_ok=True)
    
    subfolder_name = image_name.split('_')[0]  # Estrae "sub-verseXXX" da "sub-verseXXX_seg-vert_msk.nii"
    subfolder_path = os.path.join(slices_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Scorre tutte le slice e salva ciascuna nell'ordine specificato
    for slice_data, slice_name in slices:
        # Costruisce il percorso completo del file
        slices_path = os.path.join(subfolder_path, f"{slice_name}.png")

        # Salva la slice come immagine PNG
        plt.imsave(slices_path, slice_data, cmap="gray")
        print(f"Salvato {slice_name} a {slices_path}")