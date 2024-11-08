import os
import nibabel as nib #per gestire i file .nii in python 
import gzip
import shutil
import matplotlib.pyplot as plt

# Percorso principale contenente le cartelle sub-versexxx
main_path = "/mnt/d/osfstorage-archive/dataset-verse19test/derivatives"

# Liste per memorizzare le immagini caricate e i nomi dei file
nii_images = []
image_names = []

# Itera attraverso le sottocartelle
for subdir in os.listdir(main_path):
    subdir_path = os.path.join(main_path, subdir)
    
    # Controlla se è una directory con il prefisso corretto
    if os.path.isdir(subdir_path) and subdir.startswith("sub-verse"):
        # Cerca il file .gz nella sottocartella
        for file in os.listdir(subdir_path):
            if file.endswith(".gz"):
                gz_path = os.path.join(subdir_path, file)
                # Estrai il file .nii da .gz in una posizione temporanea
                nii_filename = file.replace(".gz", ".nii")
                nii_path = os.path.join(subdir_path, nii_filename)
                
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(nii_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Carica il file .nii e aggiungilo alla lista
                nii_image = nib.load(nii_path)
                nii_images.append(nii_image)
                image_names.append(nii_filename)  # Salva il nome dell'immagine
                

# Ora nii_images contiene tutte le immagini caricate e image_names i loro nomi

index = 12

# Assicurati che l'indice sia entro i limiti
if index < len(nii_images):
    selected_image = nii_images[index]
    selected_image_data = selected_image.get_fdata()

    # Visualizza una sezione dell'immagine (qui prendiamo la sezione centrale sull'asse z)
    middle_slice = selected_image_data[:, :, selected_image_data.shape[2] // 2]

    plt.imshow(middle_slice.T, cmap="gray", origin="lower")
    plt.title(f"Immagine: {image_names[index]}")
    plt.axis("off")
    plt.show()
else:
    print("coglione")