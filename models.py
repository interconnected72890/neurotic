import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5))  # 3 input channels, 6 output channels, 5x5 kernel

        self.max_pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling

        self.conv2 = nn.Conv2d(6, 16, (5, 5))  # 6 input channels, 16 output channels, 5x5 kernel

        self.fc1 = nn.Linear(16 * 5 * 5, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.max_pool(nn.functional.leaky_relu(self.conv1(x)))
        x = self.max_pool(nn.functional.leaky_relu(self.conv2(x)))

        x = x.view(x.size(0), -1)  # Flatten layer
        x = nn.functional.leaky_relu(self.fc1(x))
        x = nn.functional.leaky_relu(self.fc2(x))
        x = nn.functional.softmax(self.fc3(x))
        return x


class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.lin1 = nn.Linear(input_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, 1)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.lin1(x))
        x = nn.functional.leaky_relu(self.lin2(x))
        x = nn.functional.leaky_relu(self.lin3(x))
        x = nn.functional.leaky_relu(self.lin4(x))
        x = nn.functional.leaky_relu(self.lin5(x))
        return x


class LinearModel_2(nn.Module):
    def __init__(self, input_size):
        super(LinearModel_2, self).__init__()
        self.lin1 = nn.Linear(input_size, 12)
        self.lin2 = nn.Linear(12, 6)
        self.lin3 = nn.Linear(6, 1)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.lin1(x))
        x = nn.functional.leaky_relu(self.lin2(x))
        x = nn.functional.leaky_relu(self.lin3(x))
        return x
