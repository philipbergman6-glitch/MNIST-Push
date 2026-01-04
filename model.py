import torch
from torch import nn

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # CONVOLUTIONAL LAYERS - Extract spatial features
        self.conv_layers = nn.Sequential(
            # Layer 1: 1 channel → 32 channels, 5x5 kernel
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),  # [1,28,28] → [32,24,24]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # [32,24,24] → [32,12,12]

            # Layer 2: 32 channels → 64 channels, 5x5 kernel
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5), # [32,12,12] → [64,8,8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                       # [64,8,8] → [64,4,4]
        )

        # FULLY CONNECTED LAYERS - Classification
        # After conv: 64 channels × 4×4 spatial = 1024 features
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.25),  # Reduced from 0.5 - less aggressive for MNIST
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # x shape: [batch, 1, 28, 28]
        x = self.conv_layers(x)        # → [batch, 64, 4, 4]
        x = x.view(x.size(0), -1)      # Flatten → [batch, 1024]
        logits = self.fc_layers(x)     # → [batch, 10]
        return logits


# This allows you to verify the structure by just running 'python model.py'
if __name__ == "__main__":
    net = MNIST_CNN()
    print(net)
    print(f"Parameters: {sum(p.nelement() for p in net.parameters())}")

