import pickle
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from single_image_slicer import get_dynamic_x_center, get_slices, save_slices
#FIXME: riuscire nella segmentazione delle singole vertebre.

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
index_to_rotate = 23

# Carica i dati dell'immagine .nii
selected_image = nii_images[index_to_rotate]
image_data = selected_image.get_fdata()
# json_file = preprocess_json_files(json_files, index_to_rotate)


#Scambio assi + rotazione
transposed = rotate_image(image_data, axes=(1,0,2))
rotated_data = np.rot90(transposed,k = -1, axes=(1,2))

# Ricrea l'immagine ruotata come oggetto NIfTI

rotated_image = nib.Nifti1Image(rotated_data, affine=selected_image.affine)
nii_images[index_to_rotate] = rotated_image

sagittal_rotated_image = rotated_data[get_dynamic_x_center(rotated_data), :, :]

# Trying to obtain slices: 
slices = get_slices(index_to_rotate, nii_images, image_names, json_files, fixed_margin=13, bias=15)
save_slices(slices, image_names[index_to_rotate],"rotated_slices_output_single")

"""
plt.figure(figsize=(12, 12))  # Modifica (larghezza, altezza) 
plt.subplot(1, 2, 1)
plt.title("Trasformata")
plt.imshow(slice_rotated_image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Originale")
plt.imshow(slice_image_data, cmap="gray")
plt.show()
"""