import os
import numpy as np

def read_box_file(file_path):
    # Read the coordinates from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        box_data = [line.strip() for line in file if line.strip()] #returns list of strings
    return box_data


def parse_box(line):
    """Converts a string line of coordinates into a list of integers."""
    return list(map(int, line.strip().split(',')))
    

def sort_based_y(box_data):
    sorted_boxes = sorted(box_data, key=lambda box: int(box[1]) )  # Sort by min y-value

    return sorted_boxes


def sort_based_x(box_data):
    sorted_boxes = sorted(box_data, key=lambda box: int(box[0]) )  # Sort by min x-value

    return sorted_boxes


def cluster_lines(sorted_boxes, y_threshold):
    """Clusters boxes into lines based on a dynamic y-coordinate threshold."""
    lines = []
    current_line = [sorted_boxes[0]]
    prev_box_y = sorted_boxes[0][1]
    for box in sorted_boxes[1:]:
        if box[1] - prev_box_y < y_threshold:
            current_line.append(box)
        else:
            # new line group starts
            lines.append(current_line)
            current_line = [box]

        prev_box_y = box[1]

    lines.append(current_line) # append the last line group
    if len(lines)>1:
        print('more than one line')

    return lines


def save_sorted_boxes(box_data, output_path):
    # Save sorted boxes to a new file
    with open(output_path, 'w', encoding='utf-8') as file:
        for box in box_data:
            file.write(str(box)[1:-1] + '\n')


def rearrange_box_file(file_path, y_threshold=15):
    box_data_raw = read_box_file(file_path) # list of strings
    box_data = [parse_box(line) for line in box_data_raw] # list of lists
    box_data = sort_based_y(box_data)
    lines = cluster_lines(box_data, y_threshold)
    sorted_box_data = []
    for line_group in lines:
        sorted_box_data += sort_based_x(line_group)

    save_sorted_boxes(sorted_box_data, file_path)
    

if __name__ == '__main__':
    file_path = 'data/char_boxes/res_content_10001_156.txt'

    rearrange_box_file(file_path)