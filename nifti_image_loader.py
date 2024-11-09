import os
import nibabel as nib  # per gestire i file .nii in Python

# Percorso principale contenente le cartelle sub-versexxx
main_path = "/mnt/d/osfstorage-archive/dataset-verse19test/derivatives"

# Liste per memorizzare le immagini caricate e i nomi dei file
nii_images = []
image_names = []

# Itera attraverso le sottocartelle
for subdir in os.listdir(main_path):
    subdir_path = os.path.join(main_path, subdir)
    
    # Controlla se il percorso subdir_path sia una directory con il prefisso corretto
    if os.path.isdir(subdir_path) and subdir.startswith("sub-verse"):
        # Cerca il file .nii nella sottocartella
        for file in os.listdir(subdir_path):  # Scorre tutti i file nella subdir_path
            if file.endswith(".nii"):
                nii_path = os.path.join(subdir_path, file)
                
                # Carica il file .nii e aggiungilo alla lista
                nii_image = nib.load(nii_path)
                nii_images.append(nii_image)
                image_names.append(file)  # Salva il nome del file

# Ora nii_images contiene tutte le immagini caricate e image_names i loro nomi
