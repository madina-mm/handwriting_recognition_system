import os
from pathlib import Path
import cv2
import numpy as np
from rearrange_box_file import rearrange_box_file

def extract_letters(text):
    # Initialize an empty list to store each letter
    letters = []

    # Iterate through each character in the string
    for char in text:
        # Check if the character is a letter
        if char.isalpha():  # This checks if the character is a letter
            letters.append(char)

    return letters


def read_boxes(file_path):
    # Read the coordinates from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        boxes = [line.strip() for line in file if line.strip()]
    return boxes


if __name__ == '__main__':
    sentence_labels_dir = Path("data/sentence_labels")
    char_boxes_dir = 'data\\char_boxes'
    letters_dir = 'data\\letters'
    images_dir = 'data\\images'

    for txt_file in sentence_labels_dir.glob("*.txt"):
        print('\ntxt file we are looking at:', txt_file)
        with txt_file.open('r', encoding='utf-8') as file:
            sentence = file.read().strip()

            letters = extract_letters(sentence)

            char_count = len(letters)

        char_boxes_file = os.path.join(char_boxes_dir, 'res_' + os.path.basename(txt_file))
        rearrange_box_file(char_boxes_file)

        boxes = read_boxes(char_boxes_file)

        char_box_count = len(boxes)

        if char_count == char_box_count:
            print('letters: \n', letters)
            # Load the original image
            image_path = os.path.join(images_dir, os.path.splitext(os.path.basename(txt_file))[0] + '.png')
            img = cv2.imread(image_path)
            img_name = os.path.basename(image_path)


            for i, line in enumerate(boxes):
                points = line.strip().split(',')
                if points == ['']:
                    continue

                points = [int(p) for p in points]  # Convert each point to an integer
                # Reshape into (4, 2) format
                box = np.array(points).reshape((4, 2))

                # The box points are expected to be in a clockwise order starting from top-left
                x_min = int(min(box[:, 0]))
                y_min = int(min(box[:, 1]))
                x_max = int(max(box[:, 0]))
                y_max = int(max(box[:, 1]))

                # Extract the character region from the image
                char_img = img[y_min:y_max, x_min:x_max]

                # Save the character image
                img_file_name, img_file_extension = os.path.splitext(img_name)

                save_dir = os.path.join(letters_dir, letters[i])

                if letters[i].isupper():
                    save_dir += '_up'

                save_path = os.path.join(save_dir, f"{img_file_name}_{i}_{letters[i]}{img_file_extension}")

                cv2.imwrite(save_path, char_img)
                print('image saved to dir', save_path, letters[i])

                
                # save_path = os.path.join(letters_dir, f"{img_file_name}_{i}_{letters[i]}{img_file_extension}")
                # cv2.imwrite(save_path, char_img)

        else:
            print('Letter count and char box count does not match')