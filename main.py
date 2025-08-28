import pandas as pd
import numpy as np

dataset = pd.read_csv("dataset/emnist-byclass-train.csv", header=None)

mapping = {}
with open("dataset/emnist-byclass-mapping.txt") as f:
    for line in f:
        label, ascii_code = line.strip().split()
        mapping[int(label)] = chr(int(ascii_code))

def label_to_char(label):
    return mapping.get(label, None)

label = dataset.iloc[:,0].values
images = dataset.iloc[:,1:].values.reshape(-1,28,28,1).astype('float32') / 255.0

import torch 

num_classes = len(np.unique(label))
labels = torch.from_numpy(label).long()
images = torch.from_numpy(images).permute(0,3,1,2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

from torchvision import transforms
from torch.utils.data import Dataset

train_transform = transforms.Compose([
    transforms.ToPILImage() ,
    transforms.RandomRotation(10) ,
    transforms.RandomAffine(0 , translate=(0.1, 0.1)) ,
    transforms.ToTensor() ,
])

test_transform = transforms.Compose([
    transforms.ToPILImage() ,
    transforms.ToTensor() ,
])

class EMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        if torch.is_tensor(img):
            img = img.numpy()
        
        # Ensure grayscale shape (H, W)
        if img.ndim == 3 and img.shape[0] == 1:  # (1, H, W) -> (H, W)
            img = img.squeeze(0)
        elif img.ndim == 3 and img.shape[-1] == 1:  # (H, W, 1) stays fine
            pass
        elif img.ndim != 2:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
    
from torch.utils.data import DataLoader

train_dataset = EMNISTDataset(x_train, y_train, transform=train_transform)
test_dataset = EMNISTDataset(x_test, y_test ,transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch.nn as nn
import torch.optim as opti
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

model = CNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = opti.Adam(model.parameters(), lr=0.001)

for epoch in range(25):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss : {total_loss:.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)
        _, predicted = torch.max(outputs.data, 1)

        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print(f"Test accuracy : {accuracy:.4f}%")

torch.save(model.state_dict(), "cnn_model.pth")