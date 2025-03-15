#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from sklearn.metrics import accuracy_score


# In[ ]:


#CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Normalize depends on R,G,B mean and str
#Random cropping and horizontal flipping and normalize
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
#Data import
traindata = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testdata = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#DataLoader
trainloader = DataLoader(traindata, batch_size=64, shuffle=True)
testloader = DataLoader(testdata, batch_size=64, shuffle=False)


# In[ ]:


#model
class BasicBlock(nn.Module): #expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10):
        super().__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# In[ ]:


#model parameter
model = ResNet().to(device)
summary(model, (3, 32, 32))


# In[ ]:


#train model
num_epochs = 150
losses = []
accuracies = []
best_model_path = 'best_model.pth'
best_train_loss = float('inf')
best_test_accuracy = 0.0
best_epoch = 0

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(trainloader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(trainloader)
    losses.append(epoch_loss)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_accuracy = accuracy_score(all_labels, all_preds)
    accuracies.append(epoch_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] => "
          f"Train Loss: {epoch_loss:.4f}, "
          f"Test Accuracy: {epoch_accuracy:.4f}")

    scheduler.step()

    #best model
    if epoch_accuracy > best_test_accuracy:
        best_test_accuracy = epoch_accuracy
        best_train_loss = epoch_loss
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'test_accuracy': best_test_accuracy,
            'train_loss': best_train_loss
        }, best_model_path)
        print(f"Model saved with Test Accuracy: {best_test_accuracy:.4f}")

print(f"\nBest Model Metrics (Epoch {best_epoch+1}/{num_epochs}) -> "
      f"Train Loss: {best_train_loss:.4f}, Test Accuracy: {best_test_accuracy:.4f}")


# In[ ]:


import pickle
file_path = r"C:\Users\ymjr1\OneDrive\Desktop\deep learning\DL project1\cifar_test_nolabel.pkl"
with open(file_path, 'rb') as file:
        data = pickle.load(file)
array_data = np.array(data[b'data'])
array_data = array_data.reshape((10000, 32, 32, 3))
first_index = data[b'ids'][0]

transformed_data = []
for img_array in array_data:
    transformed_img = transform_test(img_array)
    transformed_img = transformed_img.to('cpu')
    transformed_data.append(transformed_img)

batch_tensor = torch.stack(transformed_data)  # (N, C, H, W)
print(batch_tensor.shape)


model.eval()
with torch.no_grad():
    predict_label_orgin = model(batch_tensor)
    predict_label_value, index = torch.max(predict_label_orgin, dim=1)


# In[ ]:


ids = list(range(len(index)))  # get id for index
labels_csv = [idx.item() for idx in index] # get labels
#  DataFrame
df = pd.DataFrame({'ID': ids, 'Labels': labels_csv})
df.to_csv("prediction.csv", index=False)

