from single_image_slicer import get_slices, plot_slice, save_slices
import pickle

# Carica le immagini e i nomi dal file .pkl
with open("images_data.pkl", "rb") as f:
    nii_images, image_names = pickle.load(f)

# FIXME: mancano all'appello 108,119,236 + il ciclo for che sembra arrestarsi sempre alla 108/119

for i in range(31,len(nii_images)): 
    slices = get_slices(i, nii_images, image_names)
    save_slices(slices,image_names[i])
