import torch.nn as nn
import torch.nn.functional as F
from torch import nn


class CNN_keypoint(nn.Module):

    # Defining the Constructor
    def __init__(self, num_classes):
        super(CNN_keypoint, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=1, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(in_features=96, out_features=num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x):

        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.flatten(start_dim=1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        x = self.activation(x)
        return x


class MLP(nn.Module):

    # Defining the Constructor
    def __init__(self, num_classes):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(39, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        x = self.activation(x)
        return x
