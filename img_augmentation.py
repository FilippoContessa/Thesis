import os
from PIL import Image, ImageOps
from torchvision import transforms
import shutil

input_folder = "slices_output_single"  
output_folder = "augmented_slices"
os.makedirs(output_folder, exist_ok=True)
TARGET_SIZE = (96, 96)

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

# Processa tutte le immagini nel database
for root, _, files in os.walk(input_folder):

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

    for file_name in files:
        if file_name.startswith("sub-verse"):
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
            vertebra_folder = os.path.join(sub_verse_folder, vertebra_name)
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
