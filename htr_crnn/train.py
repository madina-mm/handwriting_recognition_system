import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import get_datasets, custom_collate_fn
from models.crnn_model import CRNN
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid

def validate(model, device, val_loader, criterion, writer):
    model.eval()

    tq = tqdm(total=len(val_loader))
    tq.set_description('Validation')

    total_val_loss = 0

    with torch.no_grad():
        for data, targets, target_lengths in val_loader:
            data, targets = data.to(device), targets.to(device)
            target_lengths = target_lengths.to(device)

            output = model(data)
            output = output.permute(1, 0, 2)
            output_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=torch.long, device=device)

            val_loss = criterion(output.log_softmax(2), targets, output_lengths, target_lengths)
            total_val_loss += val_loss.item()

            tq.set_postfix(loss='%.6f' % val_loss.item())            
            tq.update(1)

            # create grid of images
            # img_grid = make_grid(data)

            # write to tensorboard
            # writer.add_image('images', img_grid)

    tq.close()

    avg_val_loss = total_val_loss / len(val_loader)

    return (avg_val_loss)
    

def train(model, device, train_loader, val_loader, optimizer, criterion, num_epochs, writer):
    
    best_avg_val_loss = 10000

    for epoch in range(num_epochs):
        model.train()

        tq = tqdm(total=len(train_loader))
        tq.set_description('Epoch %d' % (epoch+1))
        
        total_train_loss = 0

        for batch_idx, (data, targets, target_lengths) in enumerate(train_loader):
            data, targets, target_lengths = data.to(device), targets.to(device), target_lengths.to(device)
            optimizer.zero_grad()
            print(data.shape)
            output = model(data)
            # output shape needed for CTC [T, N, C] => T-sequence length, N-batch size, C-class numbers
            output = output.permute(1, 0, 2) # it was [32,249,78]  N,T,C
            output_lengths = torch.full((output.size(1),), output.size(0), dtype=torch.long, device=device)
            
            # print("Output", output.shape) # [249,32,78]
            # print("Output lengths", output_lengths.shape, output_lengths) # shape [32] tensor filled with all same value: 249 
            # print("Targets", targets.shape, targets) # encoded labels of items in batch. shape [32,44] 44 here is longest sequence, shorter ones are padded.
            # print("Target lengths", target_lengths.shape, target_lengths) # lengths of chars in each image in batch. shape [32]
            
            # Calculate loss - log_softmax(2) for log probabilities required by CTC
            loss = criterion(output.log_softmax(2), targets, output_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            tq.set_postfix(loss='%.6f' % loss.item())
            tq.update(1)
        
            # writer.add_graph(model, data)
            # writer.close()
        
        tq.close()

        avg_train_loss = total_train_loss / len(train_loader)

        print()

        avg_val_loss = validate(model, device, val_loader, criterion, writer)

        print(f'\nEpoch {epoch+1} | Loss: {avg_train_loss:.4f} | Validation loss: {avg_val_loss:.4f}\n')
        writer.add_scalar('Loss/Train', avg_train_loss, epoch+1)
        writer.add_scalar('Loss/Validate', avg_val_loss, epoch+1)

        print("Current learning rate:", optimizer.param_groups[0]['lr'])
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'])

        # if avg_val_loss < best_avg_val_loss:
        #     best_avg_val_loss = avg_val_loss
        # Save the model at each best performing epoch
        checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, f'checkpoints/model_best2.pth')
        print(f'Saving model at epoch {epoch+1}\n')


def train_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    num_epochs = 15
    batch_size = 32
    learning_rate = 0.001
    
    # Get datasets
    train_dataset, val_dataset, _ = get_datasets(
        image_folder='data/processed', 
        label_folder='data/labels', 
        train_size=0.7, 
        test_size=0.2, 
        val_size=0.1, 
        random_state=42)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn, 
        drop_last=False)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn, 
        drop_last=False)


    # Number of classes: A-Z, a-z, ÖĞƏÇŞİÜ, öğəçşüi, 0-9, space, and the blank for CTC
    num_classes = 78  # 77 visible classes + 1 blank

    # Model
    model = CRNN(num_classes=num_classes)  # Including CTC blank character
    model.to(device)

    # Optimizer and CTC Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CTCLoss(blank=0)  # Setting the CTC blank to be the last class
    
    writer = SummaryWriter('runs/exp_monitor')
    train(model, device, train_loader, val_loader, optimizer, criterion, num_epochs, writer)


if __name__ == '__main__':
    train_data()