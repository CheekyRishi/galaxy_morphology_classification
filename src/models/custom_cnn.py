import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # Convolutional blocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def custom_cnn(num_classes: int):
    return CustomCNN(num_classes)
