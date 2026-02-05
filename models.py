import torch
import torch.nn as nn
import torch.nn.functional as F

# Classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x)

# DNN Definition
class DNNEncoder(nn.Module):
    def __init__(self, input_dim, in_channels=1, hidden_dim=128, output_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.linear1.weight, std=0.01)
        nn.init.normal_(self.linear2.weight, std=0.01)
        nn.init.normal_(self.linear3.weight, std=0.01)

    def forward(self, x):
        # Extract Features
        x = x.view(x.size(0), -1)

        # Pass through MLP
        z = F.relu(self.linear1(x))
        z = self.bn1(z)
        z = F.relu(self.linear2(z))
        z = self.bn2(z)
        z = self.linear3(z)
        return F.normalize(z, dim=1)

# CNN Definition
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128, output_dim=64):
        super().__init__()

        # Convolutional layers
        # For MNIST/Fashion-MNIST (28x28) and CIFAR-10 (32x32)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)

        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Projection head
        self.fc1 = nn.Linear(128 * 4 * 4, hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Conv block 1: 28x28 or 32x32 -> 14x14 or 16x16
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv block 2: 14x14 or 16x16 -> 7x7 or 8x8
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv block 3: 7x7 or 8x8 -> 3x3 or 4x4
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Adaptive pooling to 4x4
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        # Projection head
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.normalize(x, dim=-1)

