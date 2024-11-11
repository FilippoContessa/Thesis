from single_image_slicer import get_slices, plot_slice, save_slices
import pickle

# Carica le immagini e i nomi dal file .pkl
with open("images_data.pkl", "rb") as f:
    nii_images, image_names = pickle.load(f)

    slices = get_slices(0, nii_images, image_names)
    save_slices(slices, image_names)
