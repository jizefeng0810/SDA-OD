import torch
import math
from torch import nn
import torch.nn.functional as F

class DA_Img(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DA_Img, self).__init__()

        self.conv1_da = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)
        self.bn1_da = nn.BatchNorm2d(256, momentum=0.1)
        self.conv2_da = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.bn2_da = nn.BatchNorm2d(128, momentum=0.1)
        self.conv3_da = nn.Conv2d(128, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.bn1_da, self.conv2_da, self.bn2_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.1)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.bn1_da(self.conv1_da(x)))
        x = F.leaky_relu(self.bn2_da(self.conv2_da(x)))
        img_features = self.conv3_da(x)
        img_features = img_features.view(-1, 1)
        return img_features

class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x