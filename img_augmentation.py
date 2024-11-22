import os
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Percorso dell'immagine originale
image_path = "slices_output/sub-verse004/sub-verse004_vertebra_16.png"
image = Image.open(image_path).convert("RGB")

# Percorso della cartella di output
output_folder = "transformed_images"
os.makedirs(output_folder, exist_ok=True)

# 1. Rotazione di 45 gradi
rotated_image = F.rotate(image, angle=45)
rotated_image.save(os.path.join(output_folder, "rotated_image.png"))

# 2. Scaling (ritaglio ridimensionato a 256x256)
scaled_image = F.resized_crop(image, top=0, left=0, height=image.height, width=image.width, size=(256, 256))
scaled_image.save(os.path.join(output_folder, "scaled_image.png"))

# 3. Flipping orizzontale
flipped_image = F.hflip(image)
flipped_image.save(os.path.join(output_folder, "flipped_image.png"))

# Visualizza tutte le immagini
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# Immagine originale
axes[0].imshow(image)
axes[0].set_title("Originale")
axes[0].axis("off")

# Immagine ruotata
axes[1].imshow(rotated_image)
axes[1].set_title("Ruotata")
axes[1].axis("off")

# Immagine scalata
axes[2].imshow(scaled_image)
axes[2].set_title("Scalata")
axes[2].axis("off")

# Immagine capovolta
axes[3].imshow(flipped_image)
axes[3].set_title("Capovolta")
axes[3].axis("off")

plt.tight_layout()
plt.show()