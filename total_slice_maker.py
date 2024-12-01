from single_image_slicer import get_slices, plot_slice, save_slices
import pickle
import psutil
import os
import gc

# Carica le immagini e i nomi e le info .json dai file .pkl
with open("training_images_data.pkl", "rb") as f:
    nii_images, image_names = pickle.load(f)

with open("training_json_data.pkl", "rb") as f:
    json_files = pickle.load(f)

def print_memory_usage(phase=""):
    """Stampa l'occupazione della memoria corrente e la memoria totale disponibile."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info().rss / (1024 ** 2)  # Memoria utilizzata in MB
    total_memory = psutil.virtual_memory().total / (1024 ** 2)  # Memoria totale in MB
    available_memory = psutil.virtual_memory().available / (1024 ** 2)  # Memoria disponibile in MB
    print(f"[{phase}] Memoria utilizzata: {memory_info:.2f} MB | "
          f"Memoria disponibile: {available_memory:.2f} MB | "
          f"Memoria totale: {total_memory:.2f} MB")


# Ciclo for con monitoraggio della memoria
for i in range(len(nii_images)):
    print_memory_usage(phase=f"Inizio ciclo {i}")
    
    try:
        slices = get_slices(i, nii_images, image_names, json_files,fixed_margin=13,bias=3)  # Genera le slice
        save_slices(slices, image_names[i])  # Salva le slice come immagini
    except Exception as e:
        print(f"Errore durante il processamento dell'indice {i}: {e}")
    
    # Libera le variabili inutilizzate e forza il garbage collection

    gc.collect()
    
    print_memory_usage(phase=f"Fine ciclo {i}")