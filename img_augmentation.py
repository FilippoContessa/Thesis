import os
from PIL import Image, ImageOps
from torchvision import transforms
import shutil
import numpy as np
input_folder = "slices_output"
crop_folder = "cropped_slices"
output_folder = "augmented_slices123"
os.makedirs(output_folder, exist_ok=True)
TARGET_SIZE = (96, 96)

def crop_white_area(image_path, output_path, margin=15, threshold=200):
    """
    Ritaglia un quadratino attorno alla parte bianca dell'immagine.

    Args:
        image_path (str): Percorso dell'immagine di input.
        output_path (str): Percorso per salvare l'immagine ritagliata.
        margin (int): Margine aggiunto attorno alla parte bianca.
        threshold (int): Soglia per definire il bianco (valori superiori sono considerati bianchi).
    """
    # Carica l'immagine e convertila in scala di grigi
    img = Image.open(image_path).convert("L")

    # Converte l'immagine in un array numpy
    img_array = np.array(img)

    # Trova le coordinate dei pixel bianchi
    non_zero_coords = np.argwhere(img_array > threshold)
    
    if non_zero_coords.size == 0:
        raise ValueError("Nessuna area bianca trovata nell'immagine.")

    # Determina i limiti (bordo superiore, sinistro, inferiore, destro)
    (top, left), (bottom, right) = non_zero_coords.min(axis=0), non_zero_coords.max(axis=0)

    # Aggiungi il margine opzionale
    top = max(0, top - margin)
    left = max(0, left - margin)
    bottom = min(img_array.shape[0], bottom + margin)
    right = min(img_array.shape[1], right + margin)

    # Ritaglia l'immagine
    cropped_img = img.crop((left, top, right, bottom))

    # Salva l'immagine ritagliata
    cropped_img.save(output_path)
    return output_path

# Trasformazione per il flipping
flipping_transform = transforms.RandomHorizontalFlip(p=1)  # Flip orizzontale forzato

# Lista di angoli di rotazione e fattori di scaling
rotation_angles = [0, 45, 90, 135]
scaling_factors = [0.5, 1, 1.5]

def grayscale_to_binary(image, threshold=128):
    """
    Converte un'immagine in scala di grigi in binaria.
    """
    if image.mode != "L":
        image = image.convert("L")
    return image.point(lambda x: 255 if x > threshold else 0, mode="1")

def add_black_background(img, target_size):
    """
    Aggiunge uno sfondo nero a un'immagine per adattarla a un quadrato o a un'altra dimensione target.
    """
    w, h = img.size
    new_img = Image.new("RGB", target_size, color=(0, 0, 0))
    x_offset = (target_size[0] - w) // 2
    y_offset = (target_size[1] - h) // 2
    new_img.paste(img, (x_offset, y_offset))
    return new_img

def scale_and_add_background(img, factor, target_size=TARGET_SIZE):
    """
    Ridimensiona l'immagine in base a un fattore e aggiunge sfondo nero.
    """
    new_size = (int(img.width * factor), int(img.height * factor))
    resized_img = img.resize(new_size)
    return add_black_background(resized_img, target_size)

for root, _, files in os.walk(input_folder):
    for file_name in files:
        if file_name.startswith("Component"):
            # Percorso immagine
            image_path = os.path.join(root, file_name)

            # Percorso per la cartella sub-versexxx
            sub_verse_folder = os.path.join(crop_folder, os.path.basename(os.path.dirname(image_path)))
            os.makedirs(sub_verse_folder, exist_ok=True)

            # Salvataggio dell'immagine ritagliata
            cropped_image_path = os.path.join(sub_verse_folder, f"cropped{file_name}.png")
            try:
                crop_white_area(image_path, cropped_image_path)
            except ValueError as e:
                print(f"Errore durante il ritaglio: {e}")
                continue

            print(f"Immagine ritagliata salvata per: {file_name}")


# Processa tutte le immagini nel database
for root, _, files in os.walk(crop_folder):
    """
    sagittal_found = False
    skip_folder = False
    for file_name in files:
        if "sagittal" in file_name.lower():
            sagittal_path = os.path.join(root, file_name)
            sagittal_image = Image.open(sagittal_path)
            width, height = sagittal_image.size
            if width > height: 
                skip_folder = True
                print(f"Skipping augmentation for folder {root} due to horizontal sagittal img: {file_name}")
                break
    if skip_folder:
        continue
    """
    for file_name in files:
        if file_name.startswith("cropped"):
            # Percorso immagine
            image_path = os.path.join(root, file_name)

            original_image = Image.open(image_path).convert("RGB")

            # Estrai sub-verse e vertebra dal nome del file
            sub_verse = os.path.basename(os.path.dirname(image_path))  # Nome sottocartella (sub-versexxx)
            vertebra_name = (os.path.splitext(file_name)[0]).split("_", 1)[1]

            # Percorso per la cartella sub-versexxx
            sub_verse_folder = os.path.join(output_folder, sub_verse)
            os.makedirs(sub_verse_folder, exist_ok=True)

            # Percorso per la cartella vertebraxxx
            vertebra_folder = os.path.join(sub_verse_folder,"vertebra_" + vertebra_name)
            os.makedirs(vertebra_folder, exist_ok=True)

            # Genera tutte le combinazioni di trasformazioni
            for angle in rotation_angles:
                for scale in scaling_factors:
                    # Rotazione immagine originale
                    rotated_original = original_image.rotate(angle, expand=True)
                    scaled_original = scale_and_add_background(rotated_original, scale)
                    binary_original = grayscale_to_binary(scaled_original)  # Conversione in binario

                    # Rotazione immagine flippata
                    flipped_image = flipping_transform(original_image)
                    rotated_flipped = flipped_image.rotate(angle, expand=True)
                    scaled_flipped = scale_and_add_background(rotated_flipped, scale)
                    binary_flipped = grayscale_to_binary(scaled_flipped)  # Conversione in binario

                    # Salvataggio delle immagini
                    original_save_path = os.path.join(
                        vertebra_folder, f"original_angle{angle}_scale{scale}.png"
                    )
                    flipped_save_path = os.path.join(
                        vertebra_folder, f"flipped_angle{angle}_scale{scale}.png"
                    )
                    binary_original.save(original_save_path)
                    binary_flipped.save(flipped_save_path)

            print(f"Trasformazioni salvate per: {file_name}")
