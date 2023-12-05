import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from MobileNet import MobileNet

# Assuming MobileNet is available in torchvision or custom implementation
model = MobileNet()  # Use appropriate MobileNet version

# Custom Dataset class to handle the specific data loading from text files
class CustomDataset(datasets.ImageFolder):
    def __init__(self, txt_path, transform=None):
        self.img_labels = []
        with open(txt_path, 'r') as file:
            for line in file:
                path, label = line.strip().split(' ')
                self.img_labels.append((path, int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations for training
train_transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Assuming deformed resize is similar to a simple resize
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[104.008/255, 116.669/255, 122.675/255], std=[1, 1, 1])
])

# Define transformations for validation
val_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[104.008/255, 116.669/255, 122.675/255], std=[1, 1, 1])
])

# Create datasets
train_dataset = CustomDataset(txt_path='./0_1_2_3_4_contrast_shuffle_train.txt', transform=train_transform)
val_dataset = CustomDataset(txt_path='./0_1_2_3_4_contrast_shuffle_val.txt', transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

# Assuming the use of a CrossEntropyLoss for classification
criterion = nn.CrossEntropyLoss()

max_iter = 100000

# Training loop skeleton
for epoch in range(max_iter // len(train_loader)):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        # Training steps
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Display loss every 100 iterations
        if i % 100 == 0:
            print(f'Iteration {i}, Loss: {loss.item()}')

    # Validation step every 100 epochs
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            # Validation steps
            pass

    # Save model every 1000 epochs
    if epoch % 1000 == 0:
        torch.save(model.state_dict(), f'models/mobilenet_finetune_{epoch}.pth')