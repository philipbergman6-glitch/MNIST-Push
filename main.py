import torch
import ssl

# Temporarily disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context

from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

# --- IMPORT THE BLUEPRINT ---
from model import MNISTNet

# 1. Initialize the model using the class
model = MNISTNet()
print(f"Training Model Structure:\n{model}")

# 2. Setup Ecosystem
optimiser = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

# 3. Load Data
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_set, validation_set = random_split(train_data, [55000, 5000])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=32)

# 4. Training Loop
n_epochs = 10
print("\n--- Starting Training ---")

for epoch in range(n_epochs):
    # --- Training Phase ---
    train_losses = []
    for batch in train_loader:
        x, y = batch
        x = x.view(x.size(0), -1)  # Flatten

        # Forward
        l = model(x)
        J = loss_fn(l, y)

        # Backward & Update
        optimiser.zero_grad()
        J.backward()
        optimiser.step()

        train_losses.append(J.item())

    avg_train_loss = torch.tensor(train_losses).mean().item()

    # --- Validation Phase ---
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.view(x.size(0), -1)

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

# 3. Preprocess
flat_image = test_image.view(-1, 28 * 28)

# 4. Inference
with torch.no_grad():
    output = model(flat_image)
    predicted_label = torch.argmax(output, dim=1).item()

# 5. Visualize
plt.imshow(test_image.squeeze(), cmap='gray')
plt.title(f"Actual: {actual_label} | Predicted: {predicted_label}")
plt.axis('off')
plt.show()

# 6. Save the Brain
filename = "my_mnist_brain.pt"
torch.save(model.state_dict(), filename)
print(f"\nSUCCESS: Model parameters saved to '{filename}'")