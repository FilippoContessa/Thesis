import pickle
import numpy as np
import nibabel as nib
from single_image_slicer import get_dynamic_x_center
from matplotlib import pyplot as plt

def rotate_image(image_data, axes=(1, 2)):
    """
    Ruota un'immagine 3D lungo gli assi specificati.
    :param image_data: Array numpy 3D che rappresenta l'immagine.
    :param axes: Gli assi lungo i quali ruotare (default è 90° lungo gli assi 1 e 2).
    :return: Array numpy ruotato.
    """
    return np.transpose(image_data, axes)

#TODO: Dovrebbe andare bene, va controllata la segmentazione delle singole vertebre.

# Caricare il file .pkl
with open("training_images_data.pkl", 'rb') as f:
    data = pickle.load(f)

nii_images, image_names = data
index_to_rotate = 41

# Carica i dati dell'immagine .nii
selected_image = nii_images[index_to_rotate]
image_data = selected_image.get_fdata()


#Scambio assi
transposed = rotate_image(image_data, axes=(1,0,2))

# Ruota l'immagine
rotated_data = np.rot90(transposed,k = 1, axes=(1,2))
rotated_data_2 = np.rot90(transposed,k = -1, axes=(1,2)) 
# Ricrea l'immagine ruotata come oggetto NIfTI

rotated_image = nib.Nifti1Image(rotated_data, affine=selected_image.affine)
rotated_image_2 = nib.Nifti1Image(rotated_data_2, affine=selected_image.affine)

slice_rotated_image = rotated_data[get_dynamic_x_center(rotated_data), :, :]
slice_rotated_image_2 = rotated_data_2[get_dynamic_x_center(rotated_data_2), :, :]
slice_image_data= image_data[get_dynamic_x_center(image_data), :, :]

plt.figure(figsize=(12, 12))  # Modifica (larghezza, altezza) 
plt.subplot(1, 2, 1)
plt.title("Trasformata")
plt.imshow(slice_rotated_image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Originale")
plt.imshow(slice_image_data, cmap="gray")
plt.show()