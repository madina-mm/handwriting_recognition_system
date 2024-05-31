import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, input_height=128, lstm_hidden_size=256, lstm_layers=2):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # Output: 64x128x1000
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Output: 64x64x500

            # Second convolutional layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Output: 128x64x500
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Output: 128x32x250

            # Third convolutional layer
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # Output: 256x32x250
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # Output: 256x16x250

            # Fourth convolutional layer
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # Output: 256x16x250
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # Output: 256x8x250

            # Fifth convolutional layer
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # Output: 512x8x250
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),  # Output: 512x4x250

            # Sixth convolutional layer
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0), # Output: 512x3x249
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        # Bidirectional LSTM layers
        self.rnn = nn.LSTM(512 * 3, lstm_hidden_size, num_layers=lstm_layers, bidirectional=True, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_size*2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # Change the shape to (batch, width, channels, height)
        x = x.view(b, w, -1)  # Flatten the height and channel dimensions
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x