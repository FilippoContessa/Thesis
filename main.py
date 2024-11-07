import nibabel as nib
import matplotlib.pyplot as plt

# Carica il file .nii con percorso corretto
img = nib.load(r"/mnt/d/osfstorage-archive/dataset-verse19test/derivatives/sub-verse012/verse012_CT-sag_seg.nii")

# Ottieni i dati come array numpy
img_data = img.get_fdata()

# Visualizza una fetta centrale dell'immagine
slice_index = img_data.shape[2] // 2
plt.imshow(img_data[:, :, slice_index], cmap="gray")
plt.colorbar()
plt.title("Sezione centrale dell'immagine .nii")
plt.show()
