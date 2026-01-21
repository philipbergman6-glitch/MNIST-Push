import torch
import ssl
# Temporarily disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context

from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

# --- IMPORT THE BLUEPRINT ---
from model import  MNIST_CNN

# 1. Initialize the model using the class
model = MNIST_CNN()

# Set device (use Apple Silicon GPU if available, otherwise CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)  # Move model to GPU

print(f"Training Model Structure:\n{model}")

# 2. Setup Ecosystem
# CNN needs a smaller learning rate than fully connected networks
optimiser = optim.SGD(model.parameters(), lr=3e-3)  # Changed: lr 0.01 → 0.003, added momentum
loss_fn = nn.CrossEntropyLoss()

# 3. Load Data
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_set, validation_set = random_split(train_data, [55000, 5000])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=32)

# 4. Training Loop
n_epochs = 30  # CNN needs more epochs to converge than simple networks
print("\n--- Starting Training ---")

for epoch in range(n_epochs):
    # --- Training Phase ---
    train_losses = []
    for batch_idx, batch in enumerate(train_loader):
        x, y = batch
        x, y = x.to(device), y.to(device)  # Move data to GPU
        # DON'T flatten for CNN! Keep as [batch, 1, 28, 28]

        # Forward
        l = model(x)
        J = loss_fn(l, y)

        # Backward & Update
        optimiser.zero_grad()
        J.backward()
        optimiser.step()

        train_losses.append(J.item())

        # Print progress every 200 batches
        if (batch_idx + 1) % 200 == 0:
            current_loss = torch.tensor(train_losses[-200:]).mean().item()
            print(f'  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {current_loss:.4f}')

    avg_train_loss = torch.tensor(train_losses).mean().item()

    # --- Validation Phase ---
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)  # Move data to GPU
            # DON'T flatten for CNN! Keep as [batch, 1, 28, 28]

            l = model(x)
            J = loss_fn(l, y)
            val_losses.append(J.item())

    avg_val_loss = torch.tensor(val_losses).mean().item()

    print(f'Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

# ---------------------------------------------------------
# EVERYTHING BELOW RUNS ONLY ONCE AFTER TRAINING IS DONE
# ---------------------------------------------------------

print("\n--- Starting Inference Demonstration ---")

# 1. Get a batch from validation
data_iter = iter(val_loader)
images, labels = next(data_iter)

# 2. Pick a specific image
index = 0
test_image = images[index]
actual_label = labels[index].item()

# 3. Preprocess - Add batch dimension but keep 2D structure
# CNN expects [batch, channels, height, width]

# 4. Inference
with torch.no_grad():
    test_image_input = test_image.unsqueeze(0).to(device)  # [1, 28, 28] → [1, 1, 28, 28], move to GPU
    output = model(test_image_input)
    predicted_label = torch.argmax(output, dim=1).item()

# 5. Visualize
plt.imshow(test_image.squeeze(), cmap='gray')
plt.title(f"Actual: {actual_label} | Predicted: {predicted_label}")
plt.axis('off')
plt.show()

# 6. Save the Brain
filename = "my_mnist_brain_CNN.pt"
torch.save(model.state_dict(), filename)
print(f"\nSUCCESS: Model parameters saved to '{filename}'")
