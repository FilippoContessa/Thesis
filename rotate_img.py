import pickle
import numpy as np
import nibabel as nib

def rotate_image(image_data, axes=(1, 2)):
    """
    Ruota un'immagine 3D lungo gli assi specificati.
    :param image_data: Array numpy 3D che rappresenta l'immagine.
    :param axes: Gli assi lungo i quali ruotare (default è 90° lungo gli assi 1 e 2).
    :return: Array numpy ruotato.
    """
    return np.transpose(image_data, axes)


# Caricare il file .pkl
with open("training_images_data.pkl", 'rb') as f:
    data = pickle.load(f)

# Assumendo che il contenuto sia una lista di immagini .nii e nomi
nii_images, image_names = data

# Indice dell'immagine da ruotare
index_to_rotate = 15  # Cambia con l'indice dell'immagine specifica da ruotare

# Carica i dati dell'immagine .nii
selected_image = nii_images[index_to_rotate]
image_data = selected_image.get_fdata()

print(f"Forma dell'immagine originale: {image_data.shape}")

#Scambio assi
transposed = rotate_image(image_data, axes=(1,0,2))
# Ruota l'immagine
rotated_data = np.rot90(transposed,k = 1, axes=(1,2)) 

# Ricrea l'immagine ruotata come oggetto NIfTI
rotated_image = nib.Nifti1Image(rotated_data, affine=selected_image.affine)

# Sostituisci l'immagine ruotata nella lista
nii_images[index_to_rotate] = rotated_image

# Salva il file .pkl aggiornato
with open("rotated_images_data.pkl", 'wb') as f:
    pickle.dump((nii_images, image_names), f)

print(f"Immagine ruotata e salvata nel nuovo file .pkl")
