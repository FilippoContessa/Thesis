import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Percorso della cartella di input (database di immagini)
input_folder = "slices_output/sub-verse004"  # Cartella con le sottocartelle delle immagini

# Percorso della cartella di output (per salvare le immagini trasformate)
output_folder = "augmented_slices_try"
os.makedirs(output_folder, exist_ok=True)

# Trasformazioni
def rotate_with_expand(image, degrees):
    return image.rotate(degrees,resample=Image.BICUBIC, expand=True)
scaling_transform = transforms.Resize((128, 128))  # Scaling
flipping_transform = transforms.RandomHorizontalFlip(p=1)  # Flip orizzontale forzato

# Funzione per convertire un'immagine in binario
def to_binary_image(image, threshold=128):
    """
    Converte un'immagine in binario in base a una soglia.
    Args:
        image (PIL.Image or torch.Tensor): L'immagine da convertire.
        threshold (int): Soglia per il binarizzazione (0-255).
    Returns:
        torch.Tensor: Immagine binaria come tensore.
    """
    # Converti l'immagine in tensore se necessario
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)  # Converte in tensor normalizzato [0, 1]
    
    # Applica la soglia
    binary_image = (image > (threshold / 255.0)).float()  # Converte in binario (0 o 1)
    return binary_image

# Processa tutte le immagini nel database
for root, _, files in os.walk(input_folder):
    for file_name in files:
        if file_name.endswith((".png", ".jpg", ".jpeg")) and file_name.startswith("sub-verse"):
            # Percorso immagine
            image_path = os.path.join(root, file_name)
            image = Image.open(image_path).convert("RGB")

            # Estrai sub-verse e vertebra dal nome del file
            sub_verse = os.path.basename(os.path.dirname(image_path))  # Nome sottocartella (sub-versexxx)
            vertebra_name = (os.path.splitext(file_name)[0]).split("_", 1)[1] # [0] per eliminare l'estensione, split... per dividere il nome in base al primo _ e prendere solo la seconda parte.

            # Percorso per la cartella sub-versexxx
            sub_verse_folder = os.path.join(output_folder, sub_verse)
            os.makedirs(sub_verse_folder, exist_ok=True)

            # Percorso per la cartella vertebraxxx
            vertebra_folder = os.path.join(sub_verse_folder, vertebra_name)
            os.makedirs(vertebra_folder, exist_ok=True)

            # 1. Rotazione
            rotated_image = rotate_with_expand(image, 45)  # Rotazione di 45 gradi
            binary_image = to_binary_image(rotated_image, threshold=128)  # Applica la binarizzazione
            rotated_image_path = os.path.join(vertebra_folder, "rotation.png")
            binary_image_pil = transforms.ToPILImage()(binary_image)  # Converti in immagine PIL
            binary_image_pil.save(rotated_image_path)  # Salva direttamente come PNG

            # 2. Scaling e binarizzazione
            scaled_image = scaling_transform(image)
            binary_image = to_binary_image(scaled_image, threshold=128)  # Applica la binarizzazione

            # Salva l'immagine scalata binaria
            scaled_binary_path = os.path.join(vertebra_folder, "scaling.png")
            plt.imsave(scaled_binary_path, binary_image.squeeze().numpy(), cmap="gray")  # Salva come PNG

            # 3. Flipping
            flipped_image = flipping_transform(image)
            flipped_image_path = os.path.join(vertebra_folder, "flipping.png")
            flipped_image.save(flipped_image_path)

            # Creazione immagine di confronto
            comparison_image_path = os.path.join(sub_verse_folder, f"cmp_{vertebra_name}.png")

            # Resizing immagini trasformate per la concatenazione
            original_resized = scaling_transform(image)  # Per uniformare le dimensioni
            rotated_resized = scaling_transform(rotated_image)
            flipped_resized = scaling_transform(flipped_image)

            # Concatenazione orizzontale delle immagini
            comparison_image = Image.new(
                "RGB",
                (original_resized.width * 4, original_resized.height)  # Larghezza totale x4 immagini
            )
            comparison_image.paste(original_resized.convert("RGB"), (0, 0))  # Immagine originale
            comparison_image.paste(rotated_resized.convert("RGB"), (original_resized.width, 0))  # Rotazione
            comparison_image.paste(flipped_resized.convert("RGB"), (original_resized.width * 2, 0))  # Flipping

            # Salva immagine di confronto
            comparison_image.save(comparison_image_path)
            print(f"Trasformazioni salvate per: {file_name}")
            print(f"Immagine di confronto salvata in: {comparison_image_path}")