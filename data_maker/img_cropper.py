from PIL import Image
import os

def process_images(input_folder):
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Add/check other file formats if necessary
            image_path = os.path.join(input_folder, filename)
            with Image.open(image_path) as img:
                width, height = img.size

                # Crop or pad the image
                if height > 128:
                    # Crop the image from the top
                    img = img.crop((0, 0, 1000, 128))
                elif height < 128:
                    # Create a new image with a white background
                    new_img = Image.new('RGB', (1000, 128), (255, 255, 255))
                    new_img.paste(img, (0, 0))  # Paste the original image at the top
                    img = new_img

                # Save the processed image
                img.save(image_path)

# Usage
input_folder = 'data/images'
process_images(input_folder)
