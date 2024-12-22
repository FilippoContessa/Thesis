import os
import pickle
import matplotlib.pyplot as plt
from access_to_json_files import get_vertebrae_Y_next_distance, get_vertebrae_Y_prev_distances, get_vertebrae_coordinates
import numpy as np
import nibabel as nib
from scipy.ndimage import label, find_objects, generate_binary_structure

#TODO: I problemi riguardano le vertebre che so fatte da 2 pezzi non connessi e le immagini ruotate. Comunque andrebbero capiti meglio la roba del bias se serve / se non serve, ho i miei dubbi sinceramente. 

def get_dynamic_x_center(image_data):
    """
    Calcola il centro dinamico lungo l'asse X basato sull'intensità media.
    """
    intensity_means = np.mean(image_data, axis=(1, 2))
    return np.argmax(intensity_means)

def extract_bigger_connected_components(slice_data, min_size, connectivity):
    """
    Isola la componente connessa più grande in una maschera binaria, se supera una dimensione minima.
    Args:
        slice_data (numpy.ndarray): Slice binaria.
        min_size (int): Dimensione minima per considerare una componente come rilevante.
        connectivity (int): Connectivity per il labeling (1=4-connectivity, 2=8-connectivity).
    Returns:
        numpy.ndarray: Maschera binaria contenente solo la componente connessa più grande.
    """
    # Etichettatura delle componenti connesse
    structure = generate_binary_structure(2, connectivity)
    labeled, num_features = label(slice_data, structure=structure)
    
    if num_features == 0:  # Nessuna componente rilevata
        return np.zeros_like(slice_data, dtype=bool)

    # Conta i pixel per ogni componente
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignora lo sfondo

    # Trova l'etichetta della componente connessa più grande
    largest_component_label = component_sizes.argmax()

    # Verifica che la componente connessa più grande superi la dimensione minima
    if component_sizes[largest_component_label] >= min_size:
        # Restituisci solo la componente più grande
        return (labeled == largest_component_label).astype(bool)
    else:
        # Nessuna componente soddisfa i criteri
        return np.zeros_like(slice_data, dtype=bool)

def extract_relevant_connected_components(slice_data, min_size, connectivity):
    """
    Isola la componente connessa più grande in una maschera binaria, se supera una dimensione minima.
    Args:
        slice_data (numpy.ndarray): Slice binaria.
        min_size (int): Dimensione minima per considerare una componente come rilevante.
        connectivity (int): Connectivity per il labeling (1=4-connectivity, 2=8-connectivity).
    Returns:
        numpy.ndarray: Maschera binaria contenente solo la componente connessa più grande.
    """
    # Etichettatura delle componenti connesse
    structure = generate_binary_structure(2, connectivity)
    labeled, num_features = label(slice_data, structure=structure)
    
    if num_features == 0:  # Nessuna componente rilevata
        return np.zeros_like(slice_data, dtype=bool)

    # Conta i pixel per ogni componente
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignora lo sfondo

    # Trova l'etichetta della componente connessa più grande
    largest_component_label = component_sizes.argmax()

    # Verifica che la componente connessa più grande superi la dimensione minima
    if component_sizes[largest_component_label] >= min_size:
        # Restituisci solo la componente più grande
        return (labeled == largest_component_label).astype(bool)
    else:
        # Nessuna componente soddisfa i criteri
        return np.zeros_like(slice_data, dtype=bool)

def rotate_image(image_data, axes=(1, 2)):
    """
    Ruota un'immagine 3D lungo gli assi specificati.
    :param image_data: Array numpy 3D che rappresenta l'immagine.
    :param axes: Gli assi lungo i quali ruotare (default è 90° lungo gli assi 1 e 2).
    :return: Array numpy ruotato.
    """
    return np.transpose(image_data, axes)


def transform_image(image_data):
    transposed = rotate_image(image_data, axes=(1, 0, 2))
    rotated_data = np.rot90(transposed, k=-1, axes=(1, 2))
    return rotated_data

def get_slices(index, nii_images, image_names, json_files, fixed_margin, bias):
    selected_image = nii_images[index]
    selected_image_name = image_names[index].replace("_seg-vert_msk.nii", "")
    json_info = json_files[index]  # File .json corrispondente all'immagine selezionata
    # Ottieni i dati dell'immagine
    image_data = selected_image.get_fdata()
    print(f"Nome dell'immagine selezionata: {selected_image_name}")
    # print(f"Forma dell'immagine: {image_data.shape}") 
    slices = []
    x_center = get_dynamic_x_center(image_data)
    sagittal_slice = image_data[x_center, :, :]
    """
    margin_prev = get_vertebrae_Y_prev_distances(index, json_files)
    margin_next = get_vertebrae_Y_next_distance(index, json_files)

    for vertebrae_index in range(1, len(json_info)):
        coordinates = get_vertebrae_coordinates(index, json_files, vertebrae_index)
        vertebra_label = json_info[vertebrae_index]["label"]
        #TODO: Lavora direttamente sulla sagittal slice
        # Estrazione della regione della vertebra
        if vertebrae_index == 1:
            single_vertebrae_slice = image_data[x_center,
                                                int(round(coordinates[1])) - fixed_margin - bias:
                                                int(round(coordinates[1])) + margin_next[vertebrae_index - 2], :]
        elif vertebrae_index == len(json_info) - 1:
            single_vertebrae_slice = image_data[x_center,
                                                int(round(coordinates[1])) - margin_prev[vertebrae_index - 2] - bias:
                                                int(round(coordinates[1])) + fixed_margin + bias, :]
        else:
            single_vertebrae_slice = image_data[x_center,
                                                int(round(coordinates[1])) - margin_prev[vertebrae_index - 2] - bias:
                                                int(round(coordinates[1])) + margin_next[vertebrae_index - 2] + bias, :]

        # Isola la componente connessa più grande 
        single_vertebrae_slice = extract_single_relevant_connected_component(single_vertebrae_slice, min_size=50, connectivity=1)

        # Aggiungi le slice alla lista
        slices.append((single_vertebrae_slice, f"{selected_image_name}_vertebra_{vertebra_label}"))
    """
    
    # Estrai la sagittal slice
    sagittal_slice = image_data[x_center, :, :]

    # Applica una soglia per ottenere la maschera binaria
    binary_mask = sagittal_slice > 0  # Adatta la soglia se necessario

    # Etichettatura delle componenti connesse
    labeled_mask, num_features = label(binary_mask)

    # Aggiungi ogni componente connessa rilevante alla lista
    for component_label in range(1, num_features + 1):
        component_mask = (labeled_mask == component_label)
        if component_mask.sum() >= 50:  # Considera solo le componenti connesse maggiori di una certa dimensione
            connected_slice = sagittal_slice * component_mask
            slices.append((connected_slice, f"Component_{component_label}_{selected_image_name}"))

    slices.append((sagittal_slice, f"Sagittal_{selected_image_name}"))

    return slices

def plot_slice(slice):
    plt.imshow(slice, cmap="gray")
    plt.show()

def save_slices(slices, image_name, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    subfolder_name = image_name.split('_')[0]  # Estrae "sub-verseXXX" da "sub-verseXXX_seg-vert_msk.nii"
    subfolder_path = os.path.join(output_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Scorre tutte le slice e salva ciascuna nell'ordine specificato
    for slice_data, slice_name in slices:
        # Costruisce il percorso completo del file
        slices_path = os.path.join(subfolder_path, f"{slice_name}.png")

        plt.imsave(slices_path, slice_data, cmap="gray")
        # print(f"Salvato {slice_name} a {slices_path}")