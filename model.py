import torch
import torch.nn as nn
import torch.nn.functional as F

class SkribblCNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        super(SkribblCNN, self).__init__()
        
        #Conv -> BatchNorm -> ReLU -> MaxPool (28x28 -> 14x14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv -> BatchNorm -> ReLU -> MaxPool (14x14 -> 7x7)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 32 channels *7* 7 = 1568 flattened inputs
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # [batch, 32, 7, 7] -> [batch, 1568]
        x = torch.flatten(x, start_dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # returns raw logits [Batch, num_classes]
        
        return x