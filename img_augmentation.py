import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

image_path = "slices_output/sub-verse004/sub-verse004_vertebra_16.png"
image = Image.open(image_path).convert("RGB")


# Definisci una pipeline di trasformazioni
transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),  # Rotazione casuale fino a ±30 gradi 
    transforms.RandomHorizontalFlip(p=0.5),  # Flip orizzontale con probabilità 50%
    transforms.RandomVerticalFlip(p=0.5),    # Flip verticale con probabilità 50%
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),  # Scaling casuale con crop
    transforms.ToTensor(),  # Converti in tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizzazione
])

augmented_image = transform(image)
augmented_image_np = augmented_image.permute(1,2,0).numpy()

plt.imshow(augmented_image_np)
plt.axis("off")
plt.show()