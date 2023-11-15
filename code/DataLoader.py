import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Part 1

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEDenseNet(nn.Module):
    def __init__(self, num_classes=43):
        super(SEDenseNet, self).__init__()
        self.features = models.densenet121(pretrained=True).features
        self.se_block = SEBlock(1024)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.se_block(x)
        x = self.classifier(x)
        return x


# Part 2

import torchvision.transforms as transforms
import kornia as K

base_transforms = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor(),      
    transforms.Lambda(lambda img: K.color.rgb_to_luv(img)) 
])


augment_transforms = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: K.color.rgb_to_luv(img))  
])


# Part 3

import torch.utils.data as data
import torchvision.datasets as datasets

train_data_path = "./data/TRAIN/"

train_data_base = datasets.ImageFolder(root=train_data_path, transform=base_transforms)
train_data_augmented = datasets.ImageFolder(root=train_data_path, transform=augment_transforms)

train_data_combined = data.ConcatDataset([train_data_base, train_data_augmented])

train_size = int(0.8 * len(train_data_combined))
val_size = len(train_data_combined) - train_size
train_dataset, val_dataset = data.random_split(train_data_combined, [train_size, val_size])

BATCH_SIZE = 256 
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Part 4

import torch.optim as optim
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


import pandas as pd
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def aggregate_class_labels(data_path):
    class_labels = []

    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        if os.path.isdir(folder_path):
            csv_file = os.path.join(folder_path, f'GT-{folder_name}.csv')
            if os.path.isfile(csv_file):
                df = pd.read_csv(csv_file, sep=';', usecols=['ClassId'])
                class_labels.extend(df['ClassId'].tolist())

    return np.array(class_labels)

train_data_path = "./data/TRAIN/"

classes_training = aggregate_class_labels(train_data_path)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(classes_training), y=classes_training)
class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()


model = SEDenseNet(num_classes=43).cuda()

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss(weight=class_weights)


# Part 5

import time

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, loader, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for (images, labels) in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        acc = calculate_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(loader), epoch_acc / len(loader)

def evaluate(model, loader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for (images, labels) in loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = criterion(predictions, labels)
            acc = calculate_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(loader), epoch_acc / len(loader)

EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    end_time = time.time()

    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {val_loss:.3f}, Val. Acc: {val_acc*100:.2f}%, Time: {end_time - start_time:.2f}s')


# Part 6
import matplotlib.pyplot as plt
import os

model_save_path = './models/SEDenseNet_model.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')
