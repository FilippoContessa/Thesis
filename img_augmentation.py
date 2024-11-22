import os
from PIL import Image
from torchvision import transforms

# Percorso della cartella di input (database di immagini)
input_folder = "slices_output"  # Cartella con le sottocartelle delle immagini

# Percorso della cartella di output (per salvare le immagini trasformate)
output_folder = "augmented_slices"
os.makedirs(output_folder, exist_ok=True)

# Trasformazioni
rotation_transform = transforms.RandomRotation(degrees=(45, 45))  # Rotazione 45°
scaling_transform = transforms.Resize((256, 256))  # Scaling
flipping_transform = transforms.RandomHorizontalFlip(p=1)  # Flip orizzontale forzato

# Processa tutte le immagini nel database
for root, _, files in os.walk(input_folder):
    for file_name in files:
        if file_name.endswith((".png", ".jpg", ".jpeg")) and file_name.startswith("sub-verse"):
            # Percorso immagine
            image_path = os.path.join(root, file_name)
            image = Image.open(image_path).convert("RGB")

            # Estrai sub-verse e vertebra dal nome del file
            sub_verse = os.path.basename(os.path.dirname(image_path))  # Nome sottocartella (sub-versexxx)
            vertebra_name = os.path.splitext(file_name)[0]  # Nome file senza estensione

            # Percorso per la cartella sub-versexxx
            sub_verse_folder = os.path.join(output_folder, sub_verse)
            os.makedirs(sub_verse_folder, exist_ok=True)

            # Percorso per la cartella vertebraxxx
            vertebra_folder = os.path.join(sub_verse_folder, vertebra_name)
            os.makedirs(vertebra_folder, exist_ok=True)

            # 1. Rotazione
            rotated_image = rotation_transform(image)
            rotated_image_path = os.path.join(vertebra_folder, "rotation.png")
            rotated_image.save(rotated_image_path)

            # 2. Scaling
            scaled_image = scaling_transform(image)
            scaled_image_path = os.path.join(vertebra_folder, "scaling.png")
            scaled_image.save(scaled_image_path)

            # 3. Flipping
            flipped_image = flipping_transform(image)
            flipped_image_path = os.path.join(vertebra_folder, "flipping.png")
            flipped_image.save(flipped_image_path)

            # Creazione immagine di confronto
            comparison_image_path = os.path.join(sub_verse_folder, f"comparison_{vertebra_name}.png")

            # Resizing immagini trasformate per la concatenazione
            original_resized = scaling_transform(image)  # Per uniformare le dimensioni
            rotated_resized = scaling_transform(rotated_image)
            scaled_resized = scaling_transform(scaled_image)
            flipped_resized = scaling_transform(flipped_image)

            # Concatenazione orizzontale delle immagini
            comparison_image = Image.new(
                "RGB",
                (original_resized.width * 4, original_resized.height)  # Larghezza totale x4 immagini
            )
            comparison_image.paste(original_resized, (0, 0))  # Immagine originale
            comparison_image.paste(rotated_resized, (original_resized.width, 0))  # Rotazione
            comparison_image.paste(scaled_resized, (original_resized.width * 2, 0))  # Scaling
            comparison_image.paste(flipped_resized, (original_resized.width * 3, 0))  # Flipping

            # Salva immagine di confronto
            comparison_image.save(comparison_image_path)
            print(f"Trasformazioni salvate per: {file_name}")
            print(f"Immagine di confronto salvata in: {comparison_image_path}")