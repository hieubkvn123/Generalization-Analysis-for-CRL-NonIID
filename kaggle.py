import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=256, output_dim=128):
        super(CNNEncoder, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 128, kernel_size=3, padding=1),   # 'same' padding
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3),             # no padding (valid)
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 'same' padding
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3),             # no padding (valid)
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 'same' padding
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Conv2d(512, 512, kernel_size=3),             # no padding (valid)
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self._get_flat_size(), 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024, 512)
        )

    def _get_flat_size(self):
        # Pass a dummy tensor to infer the flattened size after conv blocks
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)  # adjust input size as needed
            out = self.features(dummy)
            return out.view(1, -1).shape[1]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return F.normalize(x, dim=-1)



# CNN Definition
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=1024, output_dim=512):
        super().__init__()

        # Convolutional layers
        # For MNIST/Fashion-MNIST (28x28) and CIFAR-10 (32x32)
        hidden_dim, output_dim = 1024, 512
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
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_fc2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_fc3 = nn.BatchNorm1d(hidden_dim)
        self.g = nn.Linear(hidden_dim, output_dim)

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
        x = self.bn_fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = F.relu(x)

        x = self.g(x)

        return F.normalize(x, dim=-1)
