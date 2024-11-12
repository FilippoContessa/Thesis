import os
import nibabel as nib
import json
import pickle

# Percorso principale contenente le cartelle sub-versexxx
main_path = "/mnt/d/osfstorage-archive/dataset-verse19test/derivatives"

# Caricamento delle immagini con nomi file:

# Liste per memorizzare le immagini caricate e i nomi dei file
nii_images = []
image_names = []

# Itera attraverso le sottocartelle per caricare le immagini
for subdir in os.listdir(main_path):
    subdir_path = os.path.join(main_path, subdir)
    if os.path.isdir(subdir_path) and subdir.startswith("sub-verse"):
        for file in os.listdir(subdir_path):
            if file.endswith(".nii"):
                nii_path = os.path.join(subdir_path, file)
                nii_image = nib.load(nii_path)
                nii_images.append(nii_image)
                image_names.append(file)

# Salva le immagini e i nomi in un file .pkl
with open("images_data.pkl", "wb") as f:
    pickle.dump((nii_images, image_names), f)

print(f"Caricate e salvate {len(nii_images)} immagini .nii")


# Caricamento dei dati JSON e salvataggio in .pkl

json_files = []

for subdir in os.listdir(main_path):
    subdir_path = os.path.join(main_path, subdir)
    if os.path.isdir(subdir_path) and subdir.startswith("sub-verse"):
        for file in os.listdir(subdir_path):
            if file.endswith(".json"):
                json_path = os.path.join(subdir_path, file)
                
                # Carica il file JSON
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    json_files.append(data)

# Salva l'intera lista `json_files` in un file .pkl
pkl_path = os.path.join(main_path, "combined_data.pkl")
with open("json_data.pkl", 'wb') as pkl_file:
    pickle.dump(json_files, pkl_file)

print(f"Caricate e salvate {len(json_files)}  file JSON")

