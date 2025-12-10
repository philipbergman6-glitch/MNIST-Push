import torch
from torch import nn

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the layers (The "Parts")
        self.layer_stack = nn.Sequential(
            nn.Linear(28 * 28, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        # Define the flow of data (The "Wiring")
        logits = self.layer_stack(x)
        return logits

# This allows you to verify the structure by just running 'python model.py'
if __name__ == "__main__":
    net = MNISTNet()
    print(net)
    print(f"Parameters: {sum(p.nelement() for p in net.parameters())}")