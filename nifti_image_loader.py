import os
import nibabel as nib
import pickle

# Percorso principale contenente le cartelle sub-versexxx
main_path = "/mnt/d/osfstorage-archive/dataset-verse19test/derivatives"

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
