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


def char_count_validator(font_name):
    sentence_labels_dir = Path(f"text_labels/{font_name}")
    char_boxes_dir = f'char_boxes/{font_name}' 

    count_valid_files = 0
    counter=0
    char_diffs = 0

    for txt_file in sentence_labels_dir.glob("*.txt"):
        counter+=1

        # print('\ntxt file we are looking at:', txt_file)
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
        else:
            char_diffs += abs(char_count-char_box_count)
            # print(txt_file, char_count,char_box_count)


    print(f'{count_valid_files}/{counter}, char diff count:{char_diffs}')


if __name__ == '__main__':
    char_count_validator('JosefKPaneuropean')