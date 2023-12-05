import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from MobileNet import MobileNet

# Define the transformations for the training and validation datasets
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.875, 0.875)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.404, 0.466, 0.482], std=[1, 1, 1])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.404, 0.466, 0.482], std=[1, 1, 1])
])

# Load the datasets
train_dataset = datasets.ImageFolder('path_to_train_dataset', transform=train_transform)
val_dataset = datasets.ImageFolder('path_to_val_dataset', transform=val_transform)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the MobileNet model
model = MobileNet()  # Assuming we are using MobileNetV2

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Training loop
max_iter = 200000
for epoch in range(max_iter):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    
    # Validation step
    if epoch % 100 == 99:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the validation images: {100 * correct / total}%')

    # Save the model checkpoint
    if epoch % 10000 == 9999:
        torch.save(model.state_dict(), f'snaps/vg320.2_saliency_prob_{epoch + 1}.pth')

print('Finished Training')