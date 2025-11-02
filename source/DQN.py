import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.conv2 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.adaptive_pool(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x