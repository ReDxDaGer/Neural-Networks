import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import random
import os
import subprocess

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.mish(self.bn1(self.fc1(x)))
        x = F.mish(self.bn2(self.fc2(x)))
        x = F.mish(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)

if not os.path.exists("mnist.pth"):
    print("Model not found, training now...")
    subprocess.run(["python3", "train.py"], check=True)

model.load_state_dict(torch.load("mnist.pth"))
model.eval()

class_names = ["0","1","2","3","4","5","6","7","8","9"]

test_image, test_label = test_data[random.randint(0, len(test_data)-1)]
test_image = test_image.to(device)

with torch.no_grad():
    output = model(test_image.unsqueeze(0))
    predict_label = output.argmax(dim=1).item()

print(f"Predicted: {class_names[predict_label]}")
print(f"Actual: {class_names[test_label]}")
