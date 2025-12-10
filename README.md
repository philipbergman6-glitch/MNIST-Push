# MNIST Digit Classification

A PyTorch-based deep learning project that trains a neural network to recognize handwritten digits from the MNIST dataset.

## Project Overview

This project implements a 3-layer fully connected neural network that achieves high accuracy on digit classification. It demonstrates the complete machine learning workflow: model definition, training, and inference.

## Project Structure

```
.
├── model.py          # Neural network architecture definition
├── main.py           # Training script
├── app.py            # Inference/prediction script
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Model Architecture

- **Input Layer**: 784 neurons (28x28 flattened images)
- **Hidden Layer 1**: 16 neurons with ReLU activation
- **Hidden Layer 2**: 16 neurons with ReLU activation
- **Output Layer**: 10 neurons (digits 0-9)

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the training script to train the neural network on MNIST data:

```bash
python main.py
```

This will:
- Download the MNIST dataset automatically (first run only)
- Train for 10 epochs
- Show training and validation loss for each epoch
- Display a test prediction with visualization
- Save the trained model as `my_mnist_brain.pt`

**Expected output**: Training loss should decrease over epochs (typically reaching ~0.3-0.5)

### Running Inference

After training, test the model on new images:

```bash
python app.py
```

This will:
- Load the trained model from `my_mnist_brain.pt`
- Make a prediction on a test image
- Display the image with actual vs. predicted labels

## Technical Details

- **Framework**: PyTorch
- **Dataset**: MNIST (60,000 training images, 10,000 test images)
- **Optimizer**: Stochastic Gradient Descent (SGD) with learning rate 0.01
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 32
- **Train/Validation Split**: 55,000 / 5,000

## Learning Resources

This project is designed for educational purposes to demonstrate:
- PyTorch model definition using `nn.Module`
- Training loops with forward and backward passes
- Model saving and loading with `state_dict()`
- Data preprocessing and batch loading
- Model evaluation and visualization

## Notes

- SSL verification is disabled for MNIST downloads (handled in `main.py`)
- The model architecture is defined once in `model.py` and imported by both training and inference scripts to ensure consistency
- Images are automatically flattened from (28, 28) to (784,) before feeding into the network

## Author

Student project for learning PyTorch and deep learning fundamentals.
