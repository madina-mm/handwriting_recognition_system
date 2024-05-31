import torch
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from dataset import get_dataset_for_predict, custom_collate_fn_predict, decode_label
from models.crnn_model import CRNN  
import os
from tqdm import tqdm
# from matplotlib import pyplot as plt
# from matplotlib import matplotlib_imshow


def greedy_decoder(outputs, blank_label=0):
    """
    Decodes the output of a CTC network using a greedy approach - by selecting the most likely character
    at each timestep and collapsing repeated characters and removing blanks.

    Args:
        outputs (torch.Tensor): The raw output from the network of shape (T, N, C) where
                                T is the timestep, N is the batch size, and C is the number
                                of classes (including the blank).
        blank_label (int): The index of the CTC blank label.

    Returns:
        list: A list of decoded sequences for each item in the batch.
    """
    # Assuming outputs are already softmaxed and of shape [T, 1, C]
    outputs = outputs.squeeze(1)  # Reduce it to [T, C] for easier handling
    
    # Get the index with the maximum probability at each timestep for each batch item
    max_indices = torch.argmax(outputs, dim=1) 
    

    decoded_sequence = []
    last_index = None
    for idx in max_indices:  # Iterate through each timestep
        # Append the index if it's not a repeat of the last one and it's not the blank label
        if idx != last_index and idx != blank_label:
            decoded_sequence.append(idx.item())
        last_index = idx
    
    return decoded_sequence



def test(model, device, test_loader, criterion, checkpoint_path, predictions_folder):
    # Load the model weights from the saved checkpoint
    
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set the model to evaluation mode

    tq = tqdm(total=len(test_loader))
    tq.set_description('Test')

    with torch.no_grad():
        for data, filename in test_loader:
            data = data.to(device)
            output = model(data)
            
            output = output.permute(1, 0, 2)  # [T, N, C] format for CTC
            # print('output shape', output.shape)

            # Decode and save/print predictions
            probabilities = torch.softmax(output, dim=2)
            decoded_prediction = greedy_decoder(probabilities, blank_label=0)  # Ensure idx_to_char is defined correctly
            text_prediction = decode_label(decoded_prediction)

            label_file = filename[0][:-3]+'.txt' # since batch size is 1.
            with open(os.path.join(predictions_folder, label_file), 'w', encoding='utf-8') as f:
                f.write(text_prediction)
         
            tq.update(1)

    tq.close()
    
    # return avg_loss

# helper functions


# def plot_classes_preds(image, label, pred):
#     '''
#     Generates matplotlib Figure using a trained network, along with images
#     and labels from a batch, that shows the network's top prediction along
#     with its probability, alongside the actual label, coloring this
#     information based on whether the prediction was correct or not.
#     '''
#     # plot the images in the batch, along with predicted and true labels
#     fig = plt.figure(figsize=(12, 48))
   
#     matplotlib_imshow(image, one_channel=True)
#     fig.set_title("{0}, {1:.1f}%\n(label: {2})".format(
#         pred,
#         label,
#         color=("green" if pred==label else "red")))
#     return fig
           
# writer.add_figure('predictions vs. actuals',
#                 plot_classes_preds(net, inputs, labels),
#                 global_step=epoch * len(trainloader) + i)

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1

    # Get dataset
    test_dataset = get_dataset_for_predict(image_folder='data/processed_test')

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn_predict)

    # Number of classes: A-Z, a-z, ÖĞƏÇŞİÜ, öğəçşüi, 0-9, space, and the blank for CTC
    num_classes = 78  # 77 visible classes + 1 blank

    # Initialize the model
    model = CRNN(num_classes=num_classes)  # Including CTC blank character
    model.to(device)

    # CTC Loss
    criterion = CTCLoss(blank=0)  # CTC blank is indexed at 0

    # Path to the saved model checkpoint
    checkpoint_path = 'checkpoints/model_best_epoch5.pth'
    predictions_folder = 'data/predictions_test'

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    # Run the test
    test(model, device, test_loader, criterion, checkpoint_path, predictions_folder)
    print(f"Completed predicting")

if __name__ == '__main__':
    test_model()




