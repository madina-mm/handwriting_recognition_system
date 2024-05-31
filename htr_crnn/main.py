from preprocess import preprocess_images

if __name__ == '__main__':
    image_folder = 'data/images'
    output_folder = 'data/processed'
    preprocess_images(image_folder, output_folder)
