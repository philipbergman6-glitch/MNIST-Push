import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- CRITICAL FIX: Import the class from model.py ---
# This ensures app.py uses the EXACT same architecture as main.py
from model import MNISTNet

# 1. Initialize the model using the class
model = MNISTNet()

# 2. Load the saved "brain"
# Now the keys (layer_stack.0.weight) will match perfectly!
model.load_state_dict(torch.load('my_mnist_brain.pt', map_location='cpu'))

# 3. Set to Evaluation Mode
model.eval()

print("✅ Model loaded successfully from model.py! Ready to predict.")

# ---------------------------------------------------------
# Inference Test
# ---------------------------------------------------------

# Load just the test data
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Pick a random image (e.g., #120)
index = 120
input_image, actual_label = test_data[index]

# Flatten the image
flat_image = input_image.view(-1, 28 * 28)

# Make the prediction
with torch.no_grad():
    output = model(flat_image)
    predicted_label = torch.argmax(output, dim=1).item()

# Show the result
print(f"Actual Number: {actual_label}")
print(f"Model Says:    {predicted_label}")

plt.imshow(input_image.squeeze(), cmap='gray')
plt.title(f"Actual: {actual_label} | Predicted: {predicted_label}")
plt.axis('off')
plt.show()