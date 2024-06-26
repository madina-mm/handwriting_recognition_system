import torch
import torch.nn as nn

# # Target are to be padded
# T = 10      # Input sequence length
# C = 6      # Number of classes (including blank)
# N = 4      # Batch size
# S = 5      # Target sequence length of longest target in batch (padding length)
# S_min = 1  # Minimum target length, for demonstration purposes
# # Initialize random batch of input vectors, for *size = (T,N,C)
# input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
# print('input', input)
# # Initialize random batch of targets (0 = blank, 1:C = classes)
# target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
# print('target', target)
# input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
# print('input_lengths',input_lengths)
# target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
# print('target_lengths', target_lengths)
# ctc_loss = nn.CTCLoss()
# loss = ctc_loss(input, target, input_lengths, target_lengths)
# print(loss)
# print(input.shape, target.shape, input_lengths.shape, target_lengths.shape)
# loss.backward()

# print('\n\nTarget are to be un-padded')
# # Target are to be un-padded
# T = 50      # Input sequence length
# C = 20      # Number of classes (including blank)
# N = 16      # Batch size
# # Initialize random batch of input vectors, for *size = (T,N,C)
# input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
# print('input', input)
# input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
# print('input_lengths',input_lengths)
# # Initialize random batch of targets (0 = blank, 1:C = classes)
# target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
# print('target_lengths', target_lengths)
# target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)
# print('target', target)
# ctc_loss = nn.CTCLoss()
# loss = ctc_loss(input, target, input_lengths, target_lengths)
# print(input.shape, target.shape, input_lengths.shape, target_lengths.shape)
# loss.backward()

# Target are to be un-padded and unbatched (effectively N=1)
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)

# Initialize random batch of input vectors, for *size = (T,C)
input = torch.randn(T, C).log_softmax(1).detach().requires_grad_()
print('input', input)

input_lengths = torch.tensor(T, dtype=torch.long)
print('input_lengths',input_lengths)

# Initialize random batch of targets (0 = blank, 1:C = classes)
target_lengths = torch.randint(low=1, high=T, size=(), dtype=torch.long)
print('target_lengths', target_lengths)

target = torch.randint(low=1, high=C, size=(target_lengths,), dtype=torch.long)
print('target', target)

ctc_loss = nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)
print(input.shape, target.shape, input_lengths.shape, target_lengths.shape)
loss.backward()