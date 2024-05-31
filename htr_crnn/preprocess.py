from torchvision import transforms
from PIL import Image, ImageOps
import os
import torch

class InvertColors():
    """Inverts the colors of a PIL Image."""
    def __call__(self, img):
        return ImageOps.invert(img)


def preprocess_images(image_folder, output_folder):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure image is in grayscale
        InvertColors(),
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize tensor values
    ])

    for img_file in os.listdir(image_folder):
        if img_file.endswith('.png'):
            img_path = os.path.join(image_folder, img_file)
            image = Image.open(img_path).convert('L')  # Convert to 'L' to ensure grayscale
            image = transform(image)
            # image.save(os.path.join(output_folder, img_file)) # to see how transformed image looks like
            
            # Save or handle image tensor
            output_path = os.path.join(output_folder, img_file.replace('.png', '.pt'))
            torch.save(image, output_path)  # Save the tensor


if __name__ == '__main__':
    image_folder = 'data/images'
    output_folder = 'data/processed'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    preprocess_images(image_folder, output_folder)