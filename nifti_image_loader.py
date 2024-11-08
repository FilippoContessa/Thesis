import os
import nibabel as nib
import gzip
import shutil

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
                
                # Rimuovi il file estratto, se non necessario mantenerlo
                os.remove(nii_path)

# Ora nii_images contiene tutte le immagini caricate e image_names i loro nomi
print(f"Caricate {len(nii_images)} immagini .nii")
print("Nomi delle immagini caricate:", image_names)
