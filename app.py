import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- CRITICAL FIX: Import the class from model.py ---
# This ensures app.py uses the EXACT same architecture as main.py
from model import MNIST_CNN  # Import the CNN architecture

# 1. Initialize the model using the class
model = MNIST_CNN()

# 2. Load the saved "brain"
# Load the CNN weights trained in main.py
model.load_state_dict(torch.load('my_mnist_brain_CNN.pt', map_location='cpu'))

# 3. Set to Evaluation Mode
model.eval()

print("✅ CNN Model loaded successfully from model.py! Ready to predict.")

# ---------------------------------------------------------
# Inference Test
# ---------------------------------------------------------

# Load just the test data
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# ---------------------------------------------------------
# PART 1: Calculate Accuracy on Full Test Set (10,000 images)
# ---------------------------------------------------------
print("\n--- Testing on full test set ---")
total = 0
correct = 0

with torch.no_grad():
    for image, label in test_data:
        # CNN needs 2D structure: add batch dimension [1, 28, 28] → [1, 1, 28, 28]
        image_input = image.unsqueeze(0)  # Keep spatial structure!

        output = model(image_input)
        predicted = torch.argmax(output, dim=1).item()

        if predicted == label:
            correct += 1
        total += 1

accuracy = (correct / total) * 100
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

# ---------------------------------------------------------
# PART 2: Visualize a Single Prediction
# ---------------------------------------------------------
print("\n--- Single image visualization ---")
index = 120
input_image, actual_label = test_data[index]

# Prepare for CNN: add batch dimension but keep 2D structure
image_input = input_image.unsqueeze(0)  # [1, 28, 28] → [1, 1, 28, 28]

# Make the prediction
with torch.no_grad():
    output = model(image_input)
    predicted_label = torch.argmax(output, dim=1).item()

# Show the result
print(f"Actual Number: {actual_label}")
print(f"Model Says:    {predicted_label}")

plt.imshow(input_image.squeeze(), cmap='gray')
plt.title(f"Actual: {actual_label} | Predicted: {predicted_label}")
plt.axis('off')
plt.show()




