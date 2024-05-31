import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import os


char_to_idx = {
    '': 0,  # Blank character for CTC
    '1': 1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '0': 10,
    ' ': 11,
    'a': 12, 'b': 13, 'c': 14, 'ç': 15, 'd': 16, 'e': 17, 'ə': 18, 'f': 19, 'g': 20, 
    'ğ': 21, 'h': 22, 'x': 23, 'ı': 24, 'i': 25, 'j': 26, 'k': 27, 'q': 28, 'l': 29, 
    'm': 30, 'n': 31, 'o': 32, 'ö': 33, 'p': 34, 'r': 35, 's': 36, 'ş': 37, 't': 38, 
    'u': 39, 'ü': 40, 'v': 41, 'w': 42, 'y': 43, 'z': 44,
    'A': 45, 'B': 46, 'C': 47, 'Ç': 48, 'D': 49, 'E': 50, 'Ə': 51, 'F': 52, 'G': 53, 
    'Ğ': 54, 'H': 55, 'X': 56, 'I': 57, 'İ': 58, 'J': 59, 'K': 60, 'Q': 61, 'L': 62, 
    'M': 63, 'N': 64, 'O': 65, 'Ö': 66, 'P': 67, 'R': 68, 'S': 69, 'Ş': 70, 'T': 71, 
    'U': 72, 'Ü': 73, 'V': 74, 'W': 75, 'Y': 76, 'Z': 77
}

# Reverse the char_to_int dictionary
# idx_to_char = {v: k for k, v in char_to_idx.items()}

idx_to_char = {
    0: '', # Blank character for CTC
    1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '0', 
    11: ' ', 
    12: 'a', 13: 'b', 14: 'c', 15: 'ç', 16: 'd', 17: 'e', 18: 'ə', 19: 'f', 20: 'g', 
    21: 'ğ', 22: 'h', 23: 'x', 24: 'ı', 25: 'i', 26: 'j', 27: 'k', 28: 'q', 29: 'l', 
    30: 'm', 31: 'n', 32: 'o', 33: 'ö', 34: 'p', 35: 'r', 36: 's', 37: 'ş', 38: 't', 
    39: 'u', 40: 'ü', 41: 'v', 42: 'w', 43: 'y', 44: 'z', 
    45: 'A', 46: 'B', 47: 'C', 48: 'Ç', 49: 'D', 50: 'E', 51: 'Ə', 52: 'F', 53: 'G',
    54: 'Ğ', 55: 'H', 56: 'X', 57: 'I', 58: 'İ', 59: 'J', 60: 'K', 61: 'Q', 62: 'L', 
    63: 'M', 64: 'N', 65: 'O', 66: 'Ö', 67: 'P', 68: 'R', 69: 'S', 70: 'Ş', 71: 'T', 
    72: 'U', 73: 'Ü', 74: 'V', 75: 'W', 76: 'Y', 77: 'Z'
}


def encode_label(text):
    not_found = [char for char in text if char not in char_to_idx]
    if not_found:
        print('These characters are not considered while encoding labels:', not_found)
    
    return [char_to_idx[char] for char in text if char in char_to_idx]


def decode_label(text):
    not_found = [char for char in text if char not in idx_to_char]
    if not_found:
        print('These values are not found in char map:', not_found)
    
    return ''.join(idx_to_char[idx] for idx in text if idx in idx_to_char)


class HandwritingDataset(Dataset):
    def __init__(self, image_files, image_folder, label_folder, transform=None, mode=None):
        """
        image_files: List of filenames of preprocessed image tensors.
        image_folder: Directory containing preprocessed image tensors.
        label_folder: Directory containing label .txt files, matching image file names.
        transform: Transformations to be applied on the images (optional).
        """
        self.image_files = image_files
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_folder, image_file)
        image = torch.load(image_path)  # Assuming images are saved as tensors
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_file)

        # Read the corresponding label from the .txt file
        with open(label_path, 'r', encoding='utf-8') as file:
            label_text = file.read().strip()

        # Encode the label text to integers
        label = encode_label(label_text)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        if self.mode == 'test':
            return image, label_tensor, len(label_tensor), label_file
        
        return image, label_tensor, len(label_tensor)


def custom_collate_fn(batch):
    """Custom collate function for handling None values in batch."""
    data = [item[0] for item in batch]  # Image tensor
    targets = [item[1] for item in batch]  # Label tensor
    target_lengths = [item[2] for item in batch]  # Lengths of each label

    data = torch.stack(data, dim=0)  # Stack all image tensors
    targets = pad_sequence(targets, batch_first=True, padding_value=0)  # Pad label tensors

    if len(batch[0]) == 4: # test mode, 4th item is label filename
        label_files = [item[3] for item in batch]
        return data, targets, torch.tensor(target_lengths), label_files

    return data, targets, torch.tensor(target_lengths)


def get_datasets(image_folder, label_folder, train_size=0.7, test_size=0.2, val_size=0.1, random_state=42):
    # we have a list of all image paths and their corresponding labels
    all_images = [f for f in os.listdir(image_folder) if f.endswith('.pt')]

    # First split to separate out the test set
    train_val_images, test_images = train_test_split(all_images, test_size=test_size, random_state=random_state)

    # Second split to separate out the train and validation sets
    custom_size = val_size/(train_size + val_size) # 0.125 x 0.80 = 0.10
    train_images, val_images = train_test_split(train_val_images, test_size=custom_size, random_state=random_state)  

    # Initialize DataLoaders
    train_dataset = HandwritingDataset(train_images, image_folder, label_folder)
    val_dataset = HandwritingDataset(val_images, image_folder, label_folder)
    test_dataset = HandwritingDataset(test_images, image_folder, label_folder, mode='test')

    return train_dataset, val_dataset, test_dataset


class PredictDataset(Dataset):
    def __init__(self, image_files, image_folder, transform=None):
        """
        image_files: List of filenames of preprocessed image tensors.
        image_folder: Directory containing preprocessed image tensors.
        transform: Transformations to be applied on the images (optional).
        """
        self.image_files = image_files
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_folder, image_file)
        image = torch.load(image_path)  # Assuming images are saved as tensors
       
        if self.transform:
            image = self.transform(image)
        
        return image, image_file

def custom_collate_fn_predict(batch):
    """Custom collate function for handling None values in batch."""
    data = [item[0] for item in batch]  # Image tensor
    image_file = [item[1] for item in batch]

    data = torch.stack(data, dim=0)  # Stack all image tensors

    return data, image_file

def get_dataset_for_predict(image_folder):
    # we have a list of all image paths and their corresponding labels
    images = [f for f in os.listdir(image_folder) if f.endswith('.pt')]

    test_dataset = PredictDataset(images, image_folder)

    return test_dataset