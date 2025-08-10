import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import random
from random import randint
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_data = datasets.MNIST(root="./data", train=True,transform=transform,download=True)
test_data = datasets.MNIST(root="./data", train=False,transform=transform,download=True)
train_loader = DataLoader(train_data, shuffle=True, batch_size = 128)
test_loader = DataLoader(test_data, shuffle=False, batch_size = 128)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28,512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256,128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128,10)

    def forward(self,x):
        x = F.mish(self.bn1(self.fc1(x)))
        x = F.mish(self.bn2(self.fc2(x)))
        x = F.mish(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr =0.001, weight_decay = 1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)
citerion = nn.CrossEntropyLoss()

if not os.path.exists("mnist.pth"):
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = citerion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(),"mnist.pth")

load_model = SimpleNN()
load_model.load_state_dict(torch.load("mnist.pth"))
load_model.eval()

class_names = ["0","1","2","3","4","5","6","7","8","9"]
test_image, test_labels = test_data[random.randint(0, len(test_data) - 1)]
with torch.no_grad():
    output = load_model(test_image.unsqueeze(0))
    predict_label = output.argmax(dim=1).item()

print(f"Predicted: {class_names[predict_label]}")
print(f"Actual: {class_names[test_labels]}")
