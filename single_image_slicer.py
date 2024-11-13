import pickle
import os
import matplotlib.pyplot as plt

def get_veretbrae_distances(index, json_files):
    json_info = json_files[index]
    coordinate_differences = []
    
    for i in range(1, len(json_info)-1):  # Ignora il primo e l'ultimo elemento per evitare errori
        x_current = json_info[i].get("X")
        y_current = json_info[i].get("Y")
        z_current = json_info[i].get("Z")
        
        x_next = json_info[i + 1].get("X")
        y_next = json_info[i + 1].get("Y")
        z_next = json_info[i + 1].get("Z")
    
        if x_current is not None and x_next is not None and y_current is not None and y_next is not None and z_current is not None and z_next is not None:
            # Calcola la differenza assoluta approssimando al valore intero più vicino per ciascuna dimensione
            diff_x = round(abs(x_next - x_current) / 2)
            diff_y = round(abs(y_next - y_current) / 2)
            diff_z = round(abs(z_next - z_current) / 2)
            
            label_current = json_info[i].get("label")
            label_next = json_info[i + 1].get("label")
            
            coordinate_differences.append((label_current, label_next, diff_x, diff_y, diff_z))

    return coordinate_differences

def get_slices(index, nii_images, image_names, json_files):
    selected_image = nii_images[index]
    selected_image_name = image_names[index]
    json_info = json_files[index] # file .json corrispondente all'immagine selezionata

    # Ottieni i dati dell'immagine
    image_data = selected_image.get_fdata()
    print(f"Nome dell'immagine selezionata: {selected_image_name}")
    print(f"Forma dell'immagine: {image_data.shape}")

    # Seleziona tre slice a diverse profondità:

    # Slice verticali:
    x_center = image_data.shape[0]//2

    sagittal_slice = image_data[x_center, :,:]  #nell'asse Z ci va un intervallo di valori dato dal file json. 

    slices = [
        (sagittal_slice, f"{selected_image_name}_slice_center"),
    ]
    return slices

def plot_slice(slice):
    plt.imshow(slice, cmap="gray")
    plt.show()

def save_slices(slices, image_name):
    slices_folder = "slices_output"
    os.makedirs(slices_folder, exist_ok=True)
    
    subfolder_name = image_name.split('_')[0]  # estrae "sub-verseXXX" da "sub-verseXXX_seg-vert_msk.nii"
    subfolder_path = os.path.join(slices_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Scorre tutte le slice e salva ciascuna nell'ordine specificato
    for slice_data, slice_name in slices:
        # Costruisce il percorso completo del file
        slices_path = os.path.join(subfolder_path, f"{slice_name}.png")

        # Salva la slice come immagine PNG
        plt.imsave(slices_path, slice_data, cmap="gray")
        print(f"Salvato {slice_name} a {slices_path}")
