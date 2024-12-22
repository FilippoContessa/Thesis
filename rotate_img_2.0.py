import pickle
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import cv2
import os
from single_image_slicer import get_dynamic_x_center

def rotate_image(image_data, axes=(1, 2)):
    """
    Ruota un'immagine 3D lungo gli assi specificati.
    :param image_data: Array numpy 3D che rappresenta l'immagine.
    :param axes: Gli assi lungo i quali ruotare (default è 90° lungo gli assi 1 e 2).
    :return: Array numpy ruotato.
    """
    return np.transpose(image_data, axes)

def preprocess_json_files(json_files, index_to_rotate):
    json_file = json_files[index_to_rotate]
    for item in json_file:
        if "X" in item and "Y" in item:
            item["X"] = -item["X"]
            item["Y"] = -item["Y"]
    return json_files

# Caricamento file .pkl
with open("training_images_data.pkl", 'rb') as f:
    data = pickle.load(f)
with open("training_json_data.pkl", 'rb') as f:
    json_files = pickle.load(f)

nii_images, image_names = data
index_to_rotate = 24

# Carica i dati dell'immagine .nii
selected_image = nii_images[index_to_rotate]
image_data = selected_image.get_fdata()

# Scambio assi + rotazione

transposed = rotate_image(image_data, axes=(1, 0, 2))
rotated_data = np.rot90(transposed, k=-1, axes=(1, 2))

# Ricrea l'immagine ruotata come oggetto NIfTI
rotated_image = nib.Nifti1Image(rotated_data, affine=selected_image.affine)
nii_images[index_to_rotate] = rotated_image

sagittal_rotated_image = rotated_data[get_dynamic_x_center(rotated_data), :, :]

# Normalizza l'immagine per convertirla in uint8
sagittal_rotated_image_norm = (sagittal_rotated_image - sagittal_rotated_image.min()) / (sagittal_rotated_image.max() - sagittal_rotated_image.min())
gray = np.uint8(sagittal_rotated_image_norm * 255)

# Soglia per ottenere un'immagine binaria
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Trovare le componenti connesse
num_labels, labels = cv2.connectedComponents(binary)

# Directory per salvare le immagini delle vertebre
output_dir = "connected_region_output"
os.makedirs(output_dir, exist_ok=True)

# Estrarre ogni componente connessa e salvarla
vertebrae_count = 0
for label in range(1, num_labels):  # Ignorare lo sfondo (label 0)
    # Creare una maschera per la componente connessa corrente
    mask = np.uint8(labels == label) * 255

    # Controllare le dimensioni per escludere rumore o oggetti troppo piccoli
    if cv2.countNonZero(mask) > 500:  # Filtra regioni piccole
        vertebrae_count += 1
        output_path = os.path.join(output_dir, f'vertebra_{vertebrae_count}.png')
        cv2.imwrite(output_path, mask)


print(f"Sono state estratte {vertebrae_count} vertebre, salvate nella directory: {output_dir}")
