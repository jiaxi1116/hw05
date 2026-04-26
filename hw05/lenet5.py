import torch
import torch.nn as nn

# LeNet-5 经典实现（适配MNIST 28×28）
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层组
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层组
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # C1 -> S2
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        # C3 -> S4
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        # 展平
        x = x.view(-1, 16 * 5 * 5)
        # F5 -> F6 -> Output
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x