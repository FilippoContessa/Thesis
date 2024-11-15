from single_image_slicer import get_slices, plot_slice, save_slices
import pickle

# Carica le immagini e i nomi e le info .json dai file .pkl
with open("images_data.pkl", "rb") as f:
    nii_images, image_names = pickle.load(f)

with open("json_data.pkl", "rb") as f:
    json_files = pickle.load(f)

# FIXME: mancano all'appello 108,119,236 + il ciclo for che sembra arrestarsi sempre alla 108/119/250(30)


for i in range(6,10): 
    slices = get_slices(i, nii_images, image_names, json_files)
    save_slices(slices,image_names[i])