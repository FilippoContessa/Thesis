import os
import nibabel as nib
import json
import pickle
import gzip
import shutil

# Percorso principale contenente le cartelle sub-versexxx per il dataset training
main_path = "/mnt/d/osfstorage-archive/dataset-verse19training/derivatives"

# Funzione per decomprimere i file .gz
def decompress_gz_files(input_folder):
    decompressed_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".gz"):
                gz_file_path = os.path.join(root, file)
                decompressed_file_path = os.path.join(root, file[:-3])  # Rimuove l'estensione .gz
                
                # Decompressione
                if not os.path.exists(decompressed_file_path):  # Evita di decomprimere di nuovo se già esiste
                    with gzip.open(gz_file_path, 'rb') as gz_file:
                        with open(decompressed_file_path, 'wb') as decompressed_file:
                            shutil.copyfileobj(gz_file, decompressed_file)
                decompressed_files.append(decompressed_file_path)
    return decompressed_files

# Decomprimi tutti i file .gz
print("Decomprimo i file .gz...")
decompressed_files = decompress_gz_files(main_path)
print(f"Decompressi {len(decompressed_files)} file .gz")

# Liste per memorizzare le immagini caricate e i nomi dei file
nii_images = []
image_names = []

# Itera attraverso le sottocartelle per caricare le immagini decompressione
print("Caricamento immagini .nii...")
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
with open("training_images_data.pkl", "wb") as f:
    pickle.dump((nii_images, image_names), f)

print(f"Caricate e salvate {len(nii_images)} immagini .nii")

# Caricamento dei dati JSON e salvataggio in .pkl
json_files = []

print("Caricamento file JSON...")
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
with open("training_json_data.pkl", 'wb') as pkl_file:
    pickle.dump(json_files, pkl_file)

print(f"Caricate e salvate {len(json_files)} file JSON")