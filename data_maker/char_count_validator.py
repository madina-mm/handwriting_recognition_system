import os
from pathlib import Path

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

    count_valid_files = 0

    for txt_file in sentence_labels_dir.glob("*.txt"):
        print('\ntxt file we are looking at:', txt_file)
        with txt_file.open('r', encoding='utf-8') as file:
            sentence = file.read().strip()

            letters = extract_letters(sentence)

            char_count = len(letters)

        char_boxes_file = os.path.join(char_boxes_dir, 'res_' + os.path.basename(txt_file))
        # rearrange_box_file(char_boxes_file)

        boxes = read_boxes(char_boxes_file)

        char_box_count = len(boxes)

        if char_count == char_box_count:
            count_valid_files += 1


    print(count_valid_files)
