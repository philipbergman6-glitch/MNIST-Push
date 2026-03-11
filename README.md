# MNIST-Push

A CNN-based handwritten digit classifier built with PyTorch, with an interactive Streamlit dashboard for exploring model predictions and drawing your own digits.

**Live demo:** [mnist-push.streamlit.app](https://mnist-push-g8ufuhpnd8bfjcnpqtdnnz.streamlit.app)

## What It Does

Trains a convolutional neural network on the MNIST dataset (70,000 handwritten digits) and serves an interactive web app where you can:

- Browse all 10,000 test images with per-digit confidence distributions
- See high-confidence, uncertain, and misclassified examples side by side
- **Draw your own digit** on a canvas and get a real-time prediction
- Download the trained model weights

## Architecture

```
Input (1×28×28)
  → Conv2d(1→32, 5×5) → ReLU → MaxPool(2×2)     → [32×12×12]
  → Conv2d(32→64, 5×5) → ReLU → MaxPool(2×2)     → [64×4×4]
  → Flatten                                         → [1024]
  → Linear(1024→128) → ReLU → Dropout(0.25)
  → Linear(128→10)                                  → logits
```

~111K parameters. Trained for 30 epochs with SGD (lr=0.003), batch size 32, on a 55K/5K train/val split.

## Project Structure

```
model.py            # CNN class definition (single source of truth)
main.py             # Training script — saves weights to my_mnist_brain_CNN.pt
app.py              # CLI inference — runs accuracy test + single-image visualization
streamlit_app.py    # Interactive dashboard — explorer, drawing canvas, QR code
requirements.txt    # Dependencies
my_mnist_brain_CNN.pt  # Trained model weights (~440 KB)
```

## Quickstart

```bash
# Clone and setup
git clone https://github.com/philipbergman6-glitch/MNIST-Push.git
cd MNIST-Push
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train the model (optional — pretrained weights included)
python main.py

# Run inference from CLI
python app.py

# Launch the dashboard
streamlit run streamlit_app.py
```

## Requirements

- Python 3.10+
- PyTorch, torchvision
- Streamlit, streamlit-drawable-canvas
- matplotlib, numpy, opencv-python-headless, Pillow, qrcode

## How the Drawing Works

The canvas input goes through MNIST-style preprocessing before prediction:

1. Crop to bounding box of the drawn stroke
2. Pad to square aspect ratio
3. Resize to 20×20, center in a 28×28 frame
4. Shift to center of mass (matching MNIST alignment)
5. Normalize pixel values to [0, 1]

This preprocessing is critical — without it, the model struggles with off-center or oddly-sized drawings.

## Dashboard Tabs

| Tab | Shows |
|-----|-------|
| **High Confidence** | Examples where the model is >99.5% sure and correct |
| **Uncertain** | Examples where confidence drops below 70% |
| **Errors** | Misclassified digits from the test set |
