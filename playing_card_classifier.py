# Oyun Kartı Sınıflandırıcı - PyTorch & EfficientNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import timm
import os

# 1. Dataset Sınıfı
class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    @property
    def classes(self):
        return self.data.classes

# 2. Transform işlemleri
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 3. Veri yolları (kendi dizinine göre güncelle!)
train_dir = 'C:/Users/ROG/Desktop/pytorch/card-image-datasetclassification/test'
val_dir = 'C:/Users/ROG/Desktop/pytorch/card-image-datasetclassification/valid'
test_dir = 'C:/Users/ROG/Desktop/pytorch/card-image-datasetclassification/test'

# 4. Dataset ve Dataloader
train_dataset = PlayingCardDataset(train_dir, transform=transform)
val_dataset = PlayingCardDataset(val_dir, transform=transform)
test_dataset = PlayingCardDataset(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 5. Model sınıfı
class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super().__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.base_model.classifier = nn.Identity()  # classifier kısmını kaldır
        self.features = self.base_model
        self.classifier = nn.Linear(1280, num_classes)  # efficientnet_b0 çıkışı 1280'dır

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 6. Eğitim hazırlığı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCardClassifier(num_classes=53).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Eğitim döngüsü
num_epochs = 5
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Doğrulama
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

# 8. Test değerlendirmesi
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"\nTest Accuracy: {test_accuracy:.2f}%")
