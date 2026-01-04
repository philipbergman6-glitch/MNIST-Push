"""
Interactive MNIST CNN Demo - Streamlit App
A comprehensive interactive presentation tool for demonstrating CNN digit recognition

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import io
from streamlit_drawable_canvas import st_canvas
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import your model
from model import MNIST_CNN

# Page configuration
st.set_page_config(
    page_title="MNIST CNN Interactive Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        color: #2ecc71;
        padding: 2rem;
        background-color: #f0f2f6;
        border-radius: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None

@st.cache_resource
def load_model():
    """Load the trained CNN model"""
    model = MNIST_CNN()
    try:
        model.load_state_dict(torch.load('my_mnist_brain_CNN.pt', map_location='cpu'))
        model.eval()
        return model, True
    except FileNotFoundError:
        return None, False

@st.cache_resource
def load_test_data():
    """Load MNIST test dataset"""
    test_data = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    return test_data

def preprocess_drawing(img_array, invert=False):
    """
    Preprocess drawn image for model input - MNIST-style preprocessing

    Args:
        img_array: numpy array from canvas
        invert: whether to invert colors (black on white → white on black)

    Returns:
        torch tensor ready for model
    """
    # Convert to PIL Image
    # Handle both grayscale (2D) and RGBA (3D) arrays
    if len(img_array.shape) == 2:
        # Grayscale image (already 2D)
        img = Image.fromarray(img_array.astype('uint8'), 'L')
    elif img_array.shape[-1] == 4:
        # RGBA image
        img = Image.fromarray(img_array.astype('uint8'), 'RGBA')
        img = img.convert('L')
    elif img_array.shape[-1] == 3:
        # RGB image
        img = Image.fromarray(img_array.astype('uint8'), 'RGB')
        img = img.convert('L')
    else:
        # Default to grayscale
        img = Image.fromarray(img_array.astype('uint8'), 'L')

    # Convert to numpy for processing
    img_array = np.array(img)

    # Invert if needed (drawing is usually black on white, MNIST is white on black)
    if invert:
        img_array = 255 - img_array

    # CRITICAL: Center the digit like MNIST does
    # Find bounding box of non-zero pixels
    rows = np.any(img_array > 30, axis=1)  # Threshold to ignore noise
    cols = np.any(img_array > 30, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop to bounding box
        cropped = img_array[rmin:rmax+1, cmin:cmax+1]

        # Get dimensions
        h, w = cropped.shape

        # Determine size for the new square image (20x20 for MNIST standard)
        # MNIST uses 20x20 for the digit, then centers in 28x28
        max_dim = max(h, w)
        scale = 20.0 / max_dim if max_dim > 20 else 1.0

        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize the cropped image
        cropped_img = Image.fromarray(cropped)
        cropped_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create a black 28x28 canvas
        final_img = np.zeros((28, 28), dtype=np.uint8)

        # Center the resized digit
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2

        final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = np.array(cropped_img)

        img_array = final_img
    else:
        # If nothing drawn, just resize to 28x28
        img = Image.fromarray(img_array)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img)

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    # Convert to tensor and add batch + channel dimensions
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)

    return img_tensor, img_array

def predict_digit(model, img_tensor):
    """Make prediction on preprocessed image"""
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        prediction = torch.argmax(output, dim=1).item()
        confidence = probabilities[prediction].item() * 100

    return prediction, confidence, probabilities.numpy()

def create_confidence_chart(probabilities):
    """Create bar chart of class probabilities"""
    fig, ax = plt.subplots(figsize=(10, 4))

    digits = list(range(10))
    ax.bar(digits, probabilities * 100, color='#1f77b4', alpha=0.7)
    ax.set_xlabel('Digit', fontsize=12)
    ax.set_ylabel('Confidence (%)', fontsize=12)
    ax.set_title('Prediction Confidence for Each Digit', fontsize=14)
    ax.set_xticks(digits)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Highlight predicted digit
    max_idx = np.argmax(probabilities)
    ax.bar(max_idx, probabilities[max_idx] * 100, color='#2ecc71', alpha=0.9)

    plt.tight_layout()
    return fig

def visualize_feature_maps(model, img_tensor, layer_idx=0):
    """Visualize feature maps from convolutional layers"""

    # Hook to capture intermediate outputs
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    # Register hook on first conv layer
    if layer_idx == 0:
        hook = model.conv_layers[0].register_forward_hook(hook_fn)
    else:
        hook = model.conv_layers[3].register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model(img_tensor)

    # Remove hook
    hook.remove()

    # Get feature maps
    feature_maps = activations[0].squeeze(0)  # Remove batch dimension

    # Plot
    num_features = min(16, feature_maps.shape[0])  # Show first 16 features
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for idx, ax in enumerate(axes.flat):
        if idx < num_features:
            ax.imshow(feature_maps[idx].cpu().numpy(), cmap='viridis')
            ax.set_title(f'Filter {idx+1}', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    layer_name = "Conv Layer 1 (32 filters)" if layer_idx == 0 else "Conv Layer 2 (64 filters)"
    fig.suptitle(f'Feature Maps from {layer_name}', fontsize=16)
    plt.tight_layout()

    return fig

def calculate_accuracy_by_digit(model, test_data, num_samples=1000):
    """Calculate per-digit accuracy on test set"""
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for i, (image, label) in enumerate(test_data):
            if i >= num_samples:
                break

            image_input = image.unsqueeze(0)
            output = model(image_input)
            prediction = torch.argmax(output, dim=1).item()

            class_total[label] += 1
            if prediction == label:
                class_correct[label] += 1

    accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                  for i in range(10)]

    return accuracies, class_correct, class_total

def calculate_confusion_matrix(model, test_data, num_samples=1000):
    """Calculate confusion matrix for the model"""
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i, (image, label) in enumerate(test_data):
            if i >= num_samples:
                break

            image_input = image.unsqueeze(0)
            output = model(image_input)
            prediction = torch.argmax(output, dim=1).item()

            all_predictions.append(prediction)
            all_labels.append(label)

    cm = confusion_matrix(all_labels, all_predictions)
    return cm, all_labels, all_predictions

def visualize_confusion_matrix(cm):
    """Create a heatmap of the confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Where Does the Model Get Confused?', fontsize=14)
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))

    plt.tight_layout()
    return fig

def find_misclassified_examples(model, test_data, num_examples=10):
    """Find examples where the model made mistakes"""
    misclassified = []

    with torch.no_grad():
        for i, (image, label) in enumerate(test_data):
            if len(misclassified) >= num_examples:
                break

            image_input = image.unsqueeze(0)
            output = model(image_input)
            probabilities = F.softmax(output, dim=1)[0]
            prediction = torch.argmax(output, dim=1).item()
            confidence = probabilities[prediction].item() * 100

            if prediction != label:
                misclassified.append({
                    'index': i,
                    'image': image,
                    'true_label': label,
                    'predicted': prediction,
                    'confidence': confidence,
                    'probabilities': probabilities.numpy()
                })

    return misclassified

def visualize_learned_filters(model, layer_idx=0):
    """Visualize the actual learned conv filters (kernels)"""
    # Get the weights from first or second conv layer
    if layer_idx == 0:
        weights = model.conv_layers[0].weight.data.cpu()  # Shape: [32, 1, 5, 5]
        num_filters = 32
        title = "Conv Layer 1: Learned 5×5 Filters (32 filters)"
    else:
        weights = model.conv_layers[3].weight.data.cpu()  # Shape: [64, 32, 5, 5]
        num_filters = 64
        # For layer 2, show only first 16 filters and average across input channels
        weights = weights[:16].mean(dim=1, keepdim=True)  # Average across 32 input channels
        title = "Conv Layer 2: Learned 5×5 Filters (showing 16 of 64)"

    # Plot filters
    if layer_idx == 0:
        n_rows, n_cols = 4, 8
        filters_to_show = 32
    else:
        n_rows, n_cols = 4, 4
        filters_to_show = 16

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))

    for idx, ax in enumerate(axes.flat):
        if idx < filters_to_show:
            filter_img = weights[idx, 0].numpy()  # Get the 5×5 kernel
            ax.imshow(filter_img, cmap='RdBu_r', interpolation='nearest')
            ax.set_title(f'F{idx+1}', fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()

    return fig

def calculate_full_test_accuracy(model, test_data, progress_callback=None):
    """Calculate accuracy on entire test set with progress updates"""
    total = 0
    correct = 0

    with torch.no_grad():
        for i, (image, label) in enumerate(test_data):
            image_input = image.unsqueeze(0)
            output = model(image_input)
            prediction = torch.argmax(output, dim=1).item()

            if prediction == label:
                correct += 1
            total += 1

            if progress_callback and i % 100 == 0:
                progress_callback(i, len(test_data))

    accuracy = (correct / total) * 100
    return accuracy, correct, total

# ========================================
# MAIN APP LAYOUT
# ========================================

# Header
st.markdown('<h1 class="main-header">🧠 MNIST CNN Interactive Demo</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
    An interactive demonstration of Convolutional Neural Networks for handwritten digit recognition
    </p>
</div>
""", unsafe_allow_html=True)

# Load model
if st.session_state.model is None:
    with st.spinner('Loading trained model...'):
        model, success = load_model()
        if success:
            st.session_state.model = model
            st.success('✅ Model loaded successfully!')
        else:
            st.error('❌ Could not load model. Make sure my_mnist_brain_CNN.pt exists.')
            st.stop()
else:
    model = st.session_state.model

# Load test data
if st.session_state.test_data is None:
    with st.spinner('Loading MNIST test data...'):
        st.session_state.test_data = load_test_data()

test_data = st.session_state.test_data

# Sidebar Navigation
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    [
        "🏠 Performance Dashboard",
        "🎨 Draw & Predict",
        "🔬 Test on MNIST Images",
        "📊 Accuracy Analysis",
        "🎯 Confusion Matrix",
        "❌ Misclassification Gallery",
        "🔍 Learned Filters",
        "🧩 Feature Maps",
        "🏗️ Architecture Overview"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
invert_colors = st.sidebar.checkbox("Invert Colors (for display)", value=False)
show_preprocessing = st.sidebar.checkbox("Show Preprocessing Steps", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 About
This interactive app demonstrates a Convolutional Neural Network
trained on the MNIST dataset.

**Model Stats:**
- Architecture: 2 Conv + 2 FC layers
- Parameters: ~108,746
- Accuracy: ~98%
- Size: 0.4 MB
""")

# ========================================
# PAGE 0: PERFORMANCE DASHBOARD
# ========================================

if page == "🏠 Performance Dashboard":
    st.header("🏠 CNN Performance Dashboard")

    st.markdown("""
    **Quick overview of your trained CNN model's performance.**
    This dashboard shows the key metrics that demonstrate your model's capabilities.
    """)

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Parameters", "~109K", "Compact!")
    with col2:
        st.metric("Architecture", "2 Conv + 2 FC", "Layers")
    with col3:
        st.metric("Model Size", "0.4 MB", "Tiny!")
    with col4:
        # Calculate or load accuracy
        if 'dashboard_accuracy' not in st.session_state:
            with st.spinner('Calculating accuracy...'):
                accuracy, correct, total = calculate_full_test_accuracy(model, test_data)
                st.session_state.dashboard_accuracy = accuracy
                st.session_state.dashboard_correct = correct
                st.session_state.dashboard_total = total

        st.metric("Test Accuracy",
                 f"{st.session_state.dashboard_accuracy:.2f}%",
                 f"{st.session_state.dashboard_correct}/{st.session_state.dashboard_total}")

    # Visual performance overview
    st.markdown("---")
    st.subheader("📈 Quick Performance Snapshot")

    # Calculate per-digit accuracy
    with st.spinner('Analyzing per-digit performance...'):
        accuracies, correct, total = calculate_accuracy_by_digit(model, test_data, 1000)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Per-digit accuracy bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        digits = list(range(10))
        bars = ax.bar(digits, accuracies, color='#1f77b4', alpha=0.7)

        # Color bars
        for i, bar in enumerate(bars):
            if accuracies[i] >= 98:
                bar.set_color('#2ecc71')
            elif accuracies[i] >= 95:
                bar.set_color('#f39c12')
            else:
                bar.set_color('#e74c3c')

        ax.set_xlabel('Digit', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Per-Digit Accuracy', fontsize=14)
        ax.set_xticks(digits)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=98, color='r', linestyle='--', alpha=0.5, label='98% target')
        ax.legend()

        plt.tight_layout()
        st.pyplot(fig)

    with col_right:
        # Show some predictions
        st.markdown("### 🎲 Random Test Samples")

        if st.button("Generate New Samples", key="dashboard_samples"):
            st.session_state.dashboard_indices = np.random.choice(len(test_data), 6, replace=False)

        if 'dashboard_indices' not in st.session_state:
            st.session_state.dashboard_indices = np.random.choice(len(test_data), 6, replace=False)

        # Show 6 predictions in 2 rows
        for row in range(2):
            cols = st.columns(3)
            for col_idx, col in enumerate(cols):
                idx = st.session_state.dashboard_indices[row * 3 + col_idx]
                image, label = test_data[idx]

                with col:
                    image_input = image.unsqueeze(0)
                    prediction, confidence, _ = predict_digit(model, image_input)

                    st.image(image.squeeze().numpy(), use_container_width=True)

                    if prediction == label:
                        st.success(f"✓ {prediction} ({confidence:.0f}%)")
                    else:
                        st.error(f"✗ {prediction} ≠ {label}")

    # Key insights
    st.markdown("---")
    st.subheader("💡 Key Insights")

    best_digit = np.argmax(accuracies)
    worst_digit = np.argmin(accuracies)
    avg_accuracy = np.mean(accuracies)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"""
        **Best Performance**
        - Digit: **{best_digit}**
        - Accuracy: **{accuracies[best_digit]:.2f}%**
        - {correct[best_digit]}/{total[best_digit]} correct
        """)

    with col2:
        st.warning(f"""
        **Needs Improvement**
        - Digit: **{worst_digit}**
        - Accuracy: **{accuracies[worst_digit]:.2f}%**
        - {correct[worst_digit]}/{total[worst_digit]} correct
        """)

    with col3:
        st.success(f"""
        **Overall Stats**
        - Avg Accuracy: **{avg_accuracy:.2f}%**
        - Digits >98%: **{sum(1 for a in accuracies if a >= 98)}**/10
        - Total Tested: **{sum(total)}** images
        """)

    # Navigation hints
    st.markdown("---")
    st.info("""
    **🎯 Explore More:**
    - See **Confusion Matrix** to understand which digits get confused
    - Check **Misclassification Gallery** to see failure cases
    - View **Learned Filters** to see what the CNN discovered during training
    """)

# ========================================
# PAGE 1: DRAW & PREDICT
# ========================================

elif page == "🎨 Draw & Predict":
    st.header("🎨 Draw a Digit and See Instant Prediction")

    st.markdown("""
    **Instructions:**
    1. Draw a digit (0-9) in the canvas below
    2. Click "Predict" to see what the model thinks
    3. Try different handwriting styles!

    **💡 Tips for Better Accuracy:**
    - Draw **BIG** - fill most of the canvas
    - Use a **thick brush** (25-40 size)
    - Draw **bold, clear strokes** (like marker, not pencil)
    - Center your digit in the canvas
    - MNIST digits are quite thick and bold!
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Drawing Canvas")

        # Canvas info
        st.info("💡 Tip: Draw large and clear for best results!")

        # Canvas size
        canvas_size = 280

        # Stroke width slider
        stroke_width = st.slider("Brush Size:", 10, 60, 35, help="Thicker is better! MNIST digits are bold.")

        # Interactive drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
            stroke_width=stroke_width,
            stroke_color="#FFFFFF" if not invert_colors else "#000000",  # White stroke on black background
            background_color="#000000" if not invert_colors else "#FFFFFF",  # Black background
            height=canvas_size,
            width=canvas_size,
            drawing_mode="freedraw",
            key="canvas",
        )

        # File uploader as alternative
        st.markdown("**Or upload an image:**")
        uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'])

        # Predict button
        predict_button = st.button("🔮 Predict!", type="primary", use_container_width=True)

    with col2:
        st.subheader("Prediction Results")

        if predict_button:
            # Get image from canvas or uploaded file
            img_array = None

            if uploaded_file is not None:
                # Use uploaded file
                uploaded_image = Image.open(uploaded_file).convert('L')
                img_array = np.array(uploaded_image)
            elif canvas_result.image_data is not None:
                # Use drawn canvas
                img_array = canvas_result.image_data

            # Check if we have an image to process
            if img_array is None:
                st.warning("⚠️ Please draw something on the canvas or upload an image first!")
            else:
                if show_preprocessing:
                    st.markdown("**Preprocessing Steps:**")
                    fig_prep, axes = plt.subplots(1, 3, figsize=(12, 4))

                    # Original
                    axes[0].imshow(img_array, cmap='gray')
                    axes[0].set_title('1. Original (280×280)')
                    axes[0].axis('off')

                    # Resized
                    img_28 = np.array(Image.fromarray(img_array).resize((28, 28)))
                    axes[1].imshow(img_28, cmap='gray')
                    axes[1].set_title('2. Resized (28×28)')
                    axes[1].axis('off')

                    # Normalized
                    img_norm = img_28 / 255.0
                    axes[2].imshow(img_norm, cmap='gray')
                    axes[2].set_title('3. Normalized [0,1]')
                    axes[2].axis('off')

                    plt.tight_layout()
                    st.pyplot(fig_prep)

                # Prepare for model
                img_tensor, processed_img = preprocess_drawing(img_array, invert=True)

                # Show what the model sees
                st.markdown("### What the Model Sees (28×28)")
                st.image(processed_img, width=140, caption="After MNIST-style preprocessing", clamp=True)
                st.caption("✓ Centered ✓ Normalized ✓ Resized to 28×28")

                # Predict
                prediction, confidence, probabilities = predict_digit(model, img_tensor)

                # Display prediction
                st.markdown(f'<div class="prediction-box">Predicted: {prediction}</div>',
                           unsafe_allow_html=True)

                # Confidence metrics
                st.markdown("### Confidence Metrics")
                col_m1, col_m2, col_m3 = st.columns(3)

                with col_m1:
                    st.metric("Predicted Digit", prediction, "")
                with col_m2:
                    st.metric("Confidence", f"{confidence:.1f}%", "")
                with col_m3:
                    second_best = probabilities.argsort()[-2]
                    st.metric("Second Choice", second_best,
                             f"{probabilities[second_best]*100:.1f}%")

                # Confidence chart
                st.markdown("### Confidence Distribution")
                fig = create_confidence_chart(probabilities)
                st.pyplot(fig)

                # Interpretation
                if confidence > 95:
                    st.success(f"🎯 Very confident this is a {prediction}!")
                elif confidence > 80:
                    st.info(f"✓ Confident this is a {prediction}")
                elif confidence > 60:
                    st.warning(f"⚠️ Somewhat uncertain, but thinks it's a {prediction}")
                else:
                    st.error(f"❓ Low confidence - ambiguous handwriting")
        else:
            st.info("👆 Draw a digit and click 'Predict' to see results!")

            # Example
            st.markdown("### Example Prediction")
            example_idx = 42
            example_img, example_label = test_data[example_idx]

            display_img = example_img.squeeze().numpy()
            if invert_colors:
                display_img = 1 - display_img

            st.image(display_img, width=140, caption=f"MNIST Example (Label: {example_label})")

# ========================================
# PAGE 2: TEST ON MNIST IMAGES
# ========================================

elif page == "🔬 Test on MNIST Images":
    st.header("🔬 Test on Real MNIST Images")

    st.markdown("""
    Select random images from the MNIST test set and see how the model performs.
    The model has **never seen** these images during training!
    """)

    # Sample selection
    num_samples = st.slider("Number of images to test:", 1, 20, 6)

    if st.button("🎲 Generate Random Samples", type="primary"):
        st.session_state.random_indices = np.random.choice(len(test_data), num_samples, replace=False)

    if 'random_indices' not in st.session_state:
        st.session_state.random_indices = np.random.choice(len(test_data), num_samples, replace=False)

    # Display predictions
    cols_per_row = 3
    indices = st.session_state.random_indices

    for i in range(0, len(indices), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            if i + j < len(indices):
                idx = indices[i + j]
                image, label = test_data[idx]

                with col:
                    # Predict
                    image_input = image.unsqueeze(0)
                    prediction, confidence, _ = predict_digit(model, image_input)

                    # Display
                    display_img = image.squeeze().numpy()
                    if invert_colors:
                        display_img = 1 - display_img

                    st.image(display_img, use_container_width=True)

                    # Result
                    is_correct = prediction == label
                    if is_correct:
                        st.success(f"✓ Predicted: {prediction} (Correct!)")
                        st.caption(f"Confidence: {confidence:.1f}%")
                    else:
                        st.error(f"✗ Predicted: {prediction} (Actual: {label})")
                        st.caption(f"Confidence: {confidence:.1f}% (Wrong!)")

    # Batch statistics
    st.markdown("---")
    st.subheader("📈 Batch Statistics")

    correct = 0
    total = len(indices)

    for idx in indices:
        image, label = test_data[idx]
        image_input = image.unsqueeze(0)
        prediction, _, _ = predict_digit(model, image_input)
        if prediction == label:
            correct += 1

    accuracy = (correct / total) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", total)
    with col2:
        st.metric("Correct", correct)
    with col3:
        st.metric("Accuracy", f"{accuracy:.1f}%")

# ========================================
# PAGE 3: ACCURACY ANALYSIS
# ========================================

elif page == "📊 Accuracy Analysis":
    st.header("📊 Detailed Accuracy Analysis")

    st.markdown("""
    Detailed performance metrics on the MNIST test set.
    This shows how well the model generalizes to unseen data.
    """)

    # Calculate full accuracy
    with st.spinner('Calculating accuracy on test set...'):
        if st.button("🔄 Calculate Full Test Accuracy (10,000 images)"):
            total = 0
            correct = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            with torch.no_grad():
                for i, (image, label) in enumerate(test_data):
                    image_input = image.unsqueeze(0)
                    output = model(image_input)
                    prediction = torch.argmax(output, dim=1).item()

                    if prediction == label:
                        correct += 1
                    total += 1

                    if i % 100 == 0:
                        progress_bar.progress(i / len(test_data))
                        status_text.text(f"Processing: {i}/{len(test_data)}")

            progress_bar.progress(1.0)
            status_text.text("Complete!")

            accuracy = (correct / total) * 100
            st.session_state.full_accuracy = accuracy
            st.session_state.full_correct = correct
            st.session_state.full_total = total

    if 'full_accuracy' in st.session_state:
        st.success(f"**Overall Accuracy: {st.session_state.full_accuracy:.2f}%** "
                  f"({st.session_state.full_correct}/{st.session_state.full_total} correct)")

    # Per-digit accuracy
    st.markdown("---")
    st.subheader("📈 Per-Digit Accuracy")

    num_samples = st.slider("Number of samples per digit:", 100, 1000, 500, 100)

    if st.button("Calculate Per-Digit Accuracy"):
        with st.spinner('Analyzing per-digit performance...'):
            accuracies, correct, total = calculate_accuracy_by_digit(model, test_data, num_samples)

            # Bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            digits = list(range(10))
            bars = ax.bar(digits, accuracies, color='#1f77b4', alpha=0.7)

            # Color bars based on accuracy
            for i, bar in enumerate(bars):
                if accuracies[i] >= 98:
                    bar.set_color('#2ecc71')  # Green for high accuracy
                elif accuracies[i] >= 95:
                    bar.set_color('#f39c12')  # Orange for medium
                else:
                    bar.set_color('#e74c3c')  # Red for low accuracy

            ax.set_xlabel('Digit', fontsize=14)
            ax.set_ylabel('Accuracy (%)', fontsize=14)
            ax.set_title(f'Accuracy by Digit (tested on {num_samples} samples)', fontsize=16)
            ax.set_xticks(digits)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=98, color='r', linestyle='--', label='Target: 98%', alpha=0.5)
            ax.legend()

            plt.tight_layout()
            st.pyplot(fig)

            # Table
            st.markdown("### Detailed Breakdown")

            import pandas as pd
            df = pd.DataFrame({
                'Digit': digits,
                'Accuracy (%)': [f"{acc:.2f}" for acc in accuracies],
                'Correct': correct,
                'Total': total
            })
            st.dataframe(df, use_container_width=True)

            # Insights
            worst_digit = np.argmin(accuracies)
            best_digit = np.argmax(accuracies)

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Best Performance:** Digit {best_digit} ({accuracies[best_digit]:.2f}%)")
            with col2:
                st.warning(f"**Needs Improvement:** Digit {worst_digit} ({accuracies[worst_digit]:.2f}%)")

# ========================================
# PAGE 4: CONFUSION MATRIX
# ========================================

elif page == "🎯 Confusion Matrix":
    st.header("🎯 Confusion Matrix Analysis")

    st.markdown("""
    **Where does the model get confused?**

    The confusion matrix shows which digits the model confuses with each other.
    - **Diagonal values** (bright): Correct predictions
    - **Off-diagonal values**: Misclassifications (which digit was confused for which)
    """)

    # Sample size selector
    num_samples = st.slider("Number of test samples to analyze:", 500, 10000, 2000, 500)

    if st.button("🔄 Calculate Confusion Matrix", type="primary"):
        with st.spinner(f'Analyzing {num_samples} test samples...'):
            cm, labels, predictions = calculate_confusion_matrix(model, test_data, num_samples)
            st.session_state.confusion_matrix = cm
            st.session_state.cm_labels = labels
            st.session_state.cm_predictions = predictions

    if 'confusion_matrix' in st.session_state:
        cm = st.session_state.confusion_matrix

        # Visualize confusion matrix
        fig = visualize_confusion_matrix(cm)
        st.pyplot(fig)

        # Analysis
        st.markdown("---")
        st.subheader("🔍 Confusion Analysis")

        # Find most confused pairs
        confusion_pairs = []
        for i in range(10):
            for j in range(10):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append((i, j, cm[i, j]))

        # Sort by count
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Most Common Confusions")
            if len(confusion_pairs) > 0:
                for i, (true_label, pred_label, count) in enumerate(confusion_pairs[:5]):
                    percentage = (count / cm[true_label].sum()) * 100
                    st.warning(f"**{true_label} → {pred_label}**: {count} times ({percentage:.1f}% of all {true_label}s)")
            else:
                st.success("No confusions found! Perfect accuracy!")

        with col2:
            st.markdown("### Per-Class Accuracy from Matrix")
            accuracies = []
            for i in range(10):
                if cm[i].sum() > 0:
                    acc = (cm[i, i] / cm[i].sum()) * 100
                    accuracies.append((i, acc, cm[i, i], cm[i].sum()))

            for digit, acc, correct, total in sorted(accuracies, key=lambda x: x[1]):
                if acc >= 98:
                    st.success(f"**Digit {digit}**: {acc:.2f}% ({correct}/{total})")
                elif acc >= 95:
                    st.info(f"**Digit {digit}**: {acc:.2f}% ({correct}/{total})")
                else:
                    st.error(f"**Digit {digit}**: {acc:.2f}% ({correct}/{total})")

# ========================================
# PAGE 5: MISCLASSIFICATION GALLERY
# ========================================

elif page == "❌ Misclassification Gallery":
    st.header("❌ When the CNN Gets It Wrong")

    st.markdown("""
    **Learn from failures!**

    This section shows examples where the model made mistakes.
    Understanding failure cases helps improve the model and reveals its limitations.
    """)

    num_examples = st.slider("Number of mistakes to find:", 5, 30, 12)

    if st.button("🔍 Find Misclassified Examples", type="primary"):
        with st.spinner('Searching for mistakes...'):
            misclassified = find_misclassified_examples(model, test_data, num_examples)
            st.session_state.misclassified = misclassified

    if 'misclassified' in st.session_state:
        misclassified = st.session_state.misclassified

        if len(misclassified) == 0:
            st.success("🎉 No misclassifications found in the first batch! Try more samples.")
        else:
            st.warning(f"Found {len(misclassified)} misclassified examples")

            # Display in grid
            cols_per_row = 4
            for i in range(0, len(misclassified), cols_per_row):
                cols = st.columns(cols_per_row)

                for j, col in enumerate(cols):
                    if i + j < len(misclassified):
                        example = misclassified[i + j]

                        with col:
                            # Show image
                            img = example['image'].squeeze().numpy()
                            st.image(img, use_container_width=True)

                            # Show prediction vs truth
                            st.error(f"**Predicted: {example['predicted']}**")
                            st.success(f"**Actual: {example['true_label']}**")
                            st.caption(f"Confidence: {example['confidence']:.1f}%")

                            # Show top 3 predictions
                            with st.expander("Top 3 predictions"):
                                probs = example['probabilities']
                                top3_idx = probs.argsort()[-3:][::-1]
                                for idx in top3_idx:
                                    st.write(f"{idx}: {probs[idx]*100:.1f}%")

            # Statistics
            st.markdown("---")
            st.subheader("📊 Misclassification Statistics")

            # Count by true label
            true_label_counts = {}
            pred_label_counts = {}
            confidence_levels = []

            for ex in misclassified:
                true_label_counts[ex['true_label']] = true_label_counts.get(ex['true_label'], 0) + 1
                pred_label_counts[ex['predicted']] = pred_label_counts.get(ex['predicted'], 0) + 1
                confidence_levels.append(ex['confidence'])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Most Commonly Misclassified**")
                sorted_true = sorted(true_label_counts.items(), key=lambda x: x[1], reverse=True)
                for label, count in sorted_true[:3]:
                    st.write(f"Digit {label}: {count} times")

            with col2:
                st.markdown("**Most Common Wrong Predictions**")
                sorted_pred = sorted(pred_label_counts.items(), key=lambda x: x[1], reverse=True)
                for label, count in sorted_pred[:3]:
                    st.write(f"Predicted as {label}: {count} times")

            with col3:
                st.markdown("**Confidence Stats**")
                avg_conf = np.mean(confidence_levels)
                max_conf = np.max(confidence_levels)
                min_conf = np.min(confidence_levels)
                st.write(f"Average: {avg_conf:.1f}%")
                st.write(f"Highest: {max_conf:.1f}%")
                st.write(f"Lowest: {min_conf:.1f}%")

            # Insight
            st.info(f"""
            **💡 Insight:**
            The model's average confidence on mistakes is {avg_conf:.1f}%.
            {"This suggests the model knows when it's uncertain!" if avg_conf < 80 else "The model is overconfident even when wrong!"}
            """)

# ========================================
# PAGE 6: LEARNED FILTERS
# ========================================

elif page == "🔍 Learned Filters":
    st.header("🔍 What Did the CNN Learn?")

    st.markdown("""
    **The magic revealed!**

    These are the actual 5×5 filters (kernels) that the CNN learned during training.
    - **Red patterns**: Negative weights (looking for dark pixels)
    - **Blue patterns**: Positive weights (looking for bright pixels)

    The network discovered these patterns automatically from data—you didn't program them!
    """)

    layer_choice = st.radio(
        "Select Convolutional Layer:",
        ["Layer 1 (32 filters - Basic Features)", "Layer 2 (64 filters - Complex Patterns)"]
    )

    layer_idx = 0 if "Layer 1" in layer_choice else 1

    if st.button("🎨 Visualize Learned Filters", type="primary"):
        with st.spinner('Rendering filters...'):
            fig = visualize_learned_filters(model, layer_idx)
            st.pyplot(fig)

    # Explanation
    st.markdown("---")
    st.subheader("🧠 What Are You Looking At?")

    if layer_idx == 0:
        st.markdown("""
        **Layer 1 Filters (32 total)**

        These 5×5 kernels are applied to the input image to detect basic features:

        - **Edge Detectors**: Some filters detect vertical edges, others horizontal
        - **Diagonal Detectors**: Filters sensitive to / and \\ patterns
        - **Blob Detectors**: Filters that respond to bright or dark regions
        - **Gradient Detectors**: Filters that find intensity changes

        **Example**: If you see a vertical pattern (blue on left, red on right), that filter
        activates strongly when it sees a vertical edge in the image (like the stem of a 7).
        """)
    else:
        st.markdown("""
        **Layer 2 Filters (64 total, showing 16)**

        These filters build on Layer 1's features to detect more complex patterns:

        - **Shape Detectors**: Combinations of edges forming curves, loops
        - **Texture Detectors**: Patterns that represent digit characteristics
        - **Composite Features**: Multi-edge patterns specific to digits

        **Note**: Layer 2 filters have 32 input channels (from Layer 1), so we average
        across those to show the overall pattern.
        """)

    # Interactive insights
    st.markdown("---")
    st.subheader("💡 Key Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Random at Start**

        Before training, these filters were just random noise—completely meaningless patterns.
        """)

    with col2:
        st.success("""
        **Learned from Data**

        After training on 55,000 examples over 15 epochs, the network discovered these
        patterns automatically using backpropagation!
        """)

    st.warning("""
    **🎯 The Big Idea:**
    You didn't tell the network to look for edges or curves. It figured out on its own
    that detecting these patterns helps classify digits. This is the power of deep learning!
    """)

# ========================================
# PAGE 7: FEATURE MAPS
# ========================================

elif page == "🧩 Feature Maps":
    st.header("🧩 What Does the Network See?")

    st.markdown("""
    Visualize what the convolutional layers detect in your input image.
    This shows how the network breaks down images into features.
    """)

    # Select test image
    st.subheader("1️⃣ Select an Input Image")

    col1, col2 = st.columns([1, 2])

    with col1:
        test_idx = st.number_input("MNIST Test Index:", 0, len(test_data)-1, 42)
        image, label = test_data[test_idx]

        display_img = image.squeeze().numpy()
        if invert_colors:
            display_img = 1 - display_img

        st.image(display_img, width=200, caption=f"Input Image (Label: {label})")

        # Predict
        image_input = image.unsqueeze(0)
        prediction, confidence, _ = predict_digit(model, image_input)

        st.metric("Prediction", prediction)
        st.metric("Confidence", f"{confidence:.1f}%")

    with col2:
        st.markdown("### 2️⃣ Feature Maps Visualization")

        layer_choice = st.radio(
            "Select Convolutional Layer:",
            ["Conv Layer 1 (32 filters, 5×5)", "Conv Layer 2 (64 filters, 5×5)"]
        )

        layer_idx = 0 if "Layer 1" in layer_choice else 1

        if st.button("🔍 Visualize Features", type="primary"):
            with st.spinner('Generating feature maps...'):
                fig = visualize_feature_maps(model, image_input, layer_idx)
                st.pyplot(fig)

            st.markdown("""
            **What am I looking at?**

            Each small image shows what one filter in the convolutional layer detected:
            - **Bright areas**: Strong activation (feature detected!)
            - **Dark areas**: No activation (feature not present)

            **Layer 1** detects simple features: edges, corners, simple curves
            **Layer 2** detects complex features: loops, strokes, digit parts
            """)

    # Architecture explanation
    st.markdown("---")
    st.subheader("🏗️ How Feature Detection Works")

    st.markdown("""
    **The Feature Hierarchy:**

    ```
    Input Image (28×28)
          ↓
    [Conv Layer 1] → Detects edges, corners
          ↓  (32 feature maps, 24×24 each)
    [Max Pool] → Downsamples to 12×12
          ↓
    [Conv Layer 2] → Detects shapes, patterns
          ↓  (64 feature maps, 8×8 each)
    [Max Pool] → Downsamples to 4×4
          ↓
    [Flatten] → 1024 features
          ↓
    [FC Layers] → "This looks like a 7"
    ```

    **Key Insight:** The network automatically learns what features to detect!
    We don't program the filters—training discovers them.
    """)

# ========================================
# PAGE 5: ARCHITECTURE OVERVIEW
# ========================================

elif page == "🏗️ Architecture Overview":
    st.header("🏗️ CNN Architecture Deep Dive")

    st.markdown("""
    Detailed breakdown of the network architecture, layer by layer.
    """)

    # Model summary
    st.subheader("📋 Model Summary")

    # Create architecture table
    import pandas as pd

    arch_data = {
        'Layer': [
            'Input',
            'Conv2d (1→32, 5×5)',
            'ReLU',
            'MaxPool2d (2×2)',
            'Conv2d (32→64, 5×5)',
            'ReLU',
            'MaxPool2d (2×2)',
            'Flatten',
            'Linear (1024→128)',
            'ReLU',
            'Dropout (p=0.25)',
            'Linear (128→10)',
            'Output'
        ],
        'Output Shape': [
            '[32, 1, 28, 28]',
            '[32, 32, 24, 24]',
            '[32, 32, 24, 24]',
            '[32, 32, 12, 12]',
            '[32, 64, 8, 8]',
            '[32, 64, 8, 8]',
            '[32, 64, 4, 4]',
            '[32, 1024]',
            '[32, 128]',
            '[32, 128]',
            '[32, 128]',
            '[32, 10]',
            '[32, 10]'
        ],
        'Parameters': [
            '0',
            '832',
            '0',
            '0',
            '51,264',
            '0',
            '0',
            '0',
            '131,200',
            '0',
            '0',
            '1,290',
            '0'
        ]
    }

    df = pd.DataFrame(arch_data)
    st.dataframe(df, use_container_width=True)

    # Total parameters
    total_params = 832 + 51264 + 131200 + 1290
    st.metric("Total Trainable Parameters", f"{total_params:,}")

    # Parameter breakdown
    st.markdown("---")
    st.subheader("📊 Parameter Distribution")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    labels = ['Conv1', 'Conv2', 'FC1', 'FC2']
    sizes = [832, 51264, 131200, 1290]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    explode = (0, 0, 0.1, 0)  # Highlight FC1

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('Parameters by Layer', fontsize=14)

    # Bar chart
    ax2.bar(labels, sizes, color=colors, alpha=0.7)
    ax2.set_ylabel('Number of Parameters', fontsize=12)
    ax2.set_title('Parameter Count Comparison', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)

    # Add values on bars
    for i, v in enumerate(sizes):
        ax2.text(i, v, f'{v:,}', ha='center', va='bottom')

    plt.tight_layout()
    st.pyplot(fig)

    st.info("**Key Observation:** The first fully connected layer (FC1) contains 55% of all parameters!")

    # Receptive field
    st.markdown("---")
    st.subheader("👁️ Receptive Field Analysis")

    st.markdown("""
    **What is the receptive field?**
    The region of the input image that affects a particular output.

    | Layer | Receptive Field | Meaning |
    |-------|----------------|---------|
    | Conv1 | 5×5 pixels | Each neuron sees a 5×5 region |
    | After Pool1 | 10×10 pixels | Pooling doubles receptive field |
    | Conv2 | 14×14 pixels | Builds on previous features |
    | After Pool2 | 28×28 pixels | Sees the ENTIRE input! |

    **By the end of conv layers, the network has a global view of the image.**
    """)

    # Code view
    st.markdown("---")
    st.subheader("💻 Model Code")

    with st.expander("Show model.py implementation"):
        st.code('''
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # CONVOLUTIONAL LAYERS
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 5),      # 1→32 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5),     # 32→64 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # FULLY CONNECTED LAYERS
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        logits = self.fc_layers(x)
        return logits
''', language='python')

# ========================================
# FOOTER
# ========================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with ❤️ using Streamlit and PyTorch</p>
    <p>CNN Model: ~108k parameters | Accuracy: ~98% | Size: 0.4 MB</p>
</div>
""", unsafe_allow_html=True)
