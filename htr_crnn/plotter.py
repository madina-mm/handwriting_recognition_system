import matplotlib.pyplot as plt

epochs = range(1, 8)
training_loss = [0.8, 0.6, 0.4, 0.3, 0.25, 0.22, 0.2]
validation_loss = [0.75, 0.65, 0.55, 0.45, 0.35, 0.3, 0.25]
training_accuracy = [68, 75, 82, 86, 90, 91, 92]
validation_accuracy = [65, 70, 76, 80, 85, 87, 89]

plt.figure(figsize=(10, 5))

# Loss plot
plt.subplot(1, 2, 1)
train_loss_plt, = plt.plot(epochs, training_loss, 'go-', label='Training Loss', color='green')
val_loss_plt, = plt.plot(epochs, validation_loss, 'ro-', label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Add text labels above points
for i, txt in enumerate(training_loss):
    plt.annotate(txt, (epochs[i], training_loss[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(validation_loss):
    plt.annotate(txt, (epochs[i], validation_loss[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xticks(epochs)  # Show all epoch ticks
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Accuracy plot
plt.subplot(1, 2, 2)
train_acc_plt, = plt.plot(epochs, training_accuracy, 'go-', label='Training Accuracy', color='green')
val_acc_plt, = plt.plot(epochs, validation_accuracy, 'ro-', label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Add text labels above points
for i, txt in enumerate(training_accuracy):
    plt.annotate(txt, (epochs[i], training_accuracy[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(validation_accuracy):
    plt.annotate(txt, (epochs[i], validation_accuracy[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xticks(epochs)  # Show all epoch ticks
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
plt.show()

