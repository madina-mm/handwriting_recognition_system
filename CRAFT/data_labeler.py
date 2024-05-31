import os
from pathlib import Path
import cv2
import numpy as np
from rearrange_box_file import rearrange_box_file
from make_letter_dirs import make_letter_dirs

def extract_letters(text):
    # Initialize an empty list to store each letter
    letters = []

    # Iterate through each character in the string
    for char in text:
        # Check if the character is a letter
        if char.isalpha() or char.isnumeric():  # This checks if the character is a letter or digit
            letters.append(char)
        

    return letters


def read_boxes(file_path):
    # Read the coordinates from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        boxes = [line.strip() for line in file if line.strip()]
    return boxes


aze_letter_conversions = {
    'ç' : 'c',
    'ə' : 'e',
    'ğ' : 'g',
    'ı' : 'i',
    'ö' : 'o',
    'ş' : 's',
    'ü' : 'u'
}


if __name__ == '__main__':
    font_name = 'AromiaScriptThin'
    sentence_labels_dir = Path("../data_maker/data/text_labels")
    char_boxes_dir = Path(f'char_boxes/{font_name}')
    letters_dir = f'letters\\{font_name}'
    images_dir = f'test_images\\{font_name}'

    make_letter_dirs(letters_dir)

    for char_boxes_file in char_boxes_dir.glob("*.txt"):
        # print('\nbox file we are looking at:', txt_file)
        rearrange_box_file(char_boxes_file)

        boxes = read_boxes(char_boxes_file)

        char_box_count = len(boxes)

        txt_file = os.path.join(sentence_labels_dir, os.path.basename(char_boxes_file)[4:])

        with open(txt_file, 'r', encoding='utf-8') as file:
            sentence = file.read().strip()

            letters = extract_letters(sentence)

            char_count = len(letters)


        if char_count == char_box_count:
            # print('letters: \n', letters)
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

                if letters[i] in 'iılt':
                    x_min += 2
                    x_max -= 2
                
                if letters[i] == 'm':
                    x_min -= 2

                # Extract the character region from the image
                char_img = img[y_min:y_max, x_min:x_max]

                # Save the character image
                img_file_name, img_file_extension = os.path.splitext(img_name)


                if letters[i].lower() in 'çəğıöşü':
                    save_dir = os.path.join(letters_dir, f'{aze_letter_conversions[letters[i].lower()]}_az')
                else:
                    save_dir = os.path.join(letters_dir, letters[i])


                if letters[i].isupper():
                    save_dir += '_up'

                save_path = os.path.join(save_dir, f"{img_file_name}_{i}{img_file_extension}")

                cv2.imwrite(save_path, char_img)
                # print('image saved to dir', save_path, letters[i])

                
                # save_path = os.path.join(letters_dir, f"{img_file_name}_{i}_{letters[i]}{img_file_extension}")
                # cv2.imwrite(save_path, char_img)


        # else:
        #     print('Letter count and char box count does not match')