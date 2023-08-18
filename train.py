import torch
import torch.nn as nn
import torch.optim as optim
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from model import EmotionClassifier
import config


train_data = CustomDataset(config.TRAIN_DIR)
train_loader = DataLoader(train_data,
                          batch_size=config.BATCH_SIZE,
                          num_workers=config.NUM_WORKER,
                          shuffle=True)

model = EmotionClassifier().to(config.DEVICE)

optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# training loop
for epoch in range(config.NUM_EPOCH):

    # total loss after one epoch
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for image, labels in train_loader:
        image, labels = image.to(config.DEVICE), labels.to(config.DEVICE)

        outputs = model(image)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get predicted labels
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    train_loss = total_loss / len(train_loader)
    epoch_accuracy = total_correct / total_samples
    print(f'Epoch [{epoch+1}/{config.NUM_EPOCH}]-Loss: {train_loss:.6f} | Accuracy: {epoch_accuracy*100:.4f}%')


# saving the model to ./models directory
torch.save(model.state_dict(), config.MODEL_PATH)
