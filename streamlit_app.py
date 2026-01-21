import streamlit as st
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import qrcode
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Can AI Read Your Handwriting?",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# BAMBOO THEME CSS
# ============================================================
st.markdown("""
    <style>
    /* Bamboo color palette */
    :root {
        --bamboo-primary: #7C9473;
        --bamboo-light: #A8B899;
        --bamboo-dark: #5F7C57;
        --bamboo-accent: #C17B6C;
        --bamboo-neutral: #E8EBE4;
        --bamboo-text: #3E4A3A;
        --bamboo-bg: #F5F7F2;
    }

    /* Bamboo style background */
    .stApp {
        background: linear-gradient(135deg, #f5f7f2 0%, #e8ebe4 50%, #f0f2ed 100%);
        background-attachment: fixed;
    }

    /* Main background */
    .main {
        background-color: transparent;
    }

    /* Headers */
    h1 {
        color: var(--bamboo-dark);
        font-weight: 300;
        letter-spacing: -1px;
        padding-bottom: 0.5rem;
        text-align: center;
        font-size: 4rem;
    }

    h2, h3 {
        color: var(--bamboo-primary);
        font-weight: 400;
        font-size: 2.5rem;
    }

    /* Make all text bigger for presentation */
    p, li, div, span, label {
        font-size: 1.4rem !important;
    }

    /* Captions larger too */
    .caption, small {
        font-size: 1.2rem !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--bamboo-dark);
        font-size: 3rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--bamboo-text);
        font-size: 1.3rem !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 1.2rem !important;
    }

    /* Dividers */
    hr {
        border-color: var(--bamboo-neutral);
        margin: 2rem 0;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--bamboo-primary);
        background-color: transparent;
        font-size: 1.4rem !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--bamboo-dark);
        border-bottom-color: var(--bamboo-dark);
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--bamboo-primary);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        border-radius: 4px;
        font-weight: 400;
        font-size: 1.4rem !important;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background-color: var(--bamboo-dark);
        border: none;
    }

    /* Download button */
    .stDownloadButton > button {
        font-size: 1.4rem !important;
    }

    /* Slider */
    .stSlider > div > div > div {
        background-color: var(--bamboo-neutral);
    }

    .stSlider [role="slider"] {
        background-color: var(--bamboo-primary);
    }

    /* Success/Error messages */
    .stSuccess {
        background-color: rgba(95, 124, 87, 0.1);
        border-left: 4px solid var(--bamboo-dark);
        color: var(--bamboo-text);
    }

    .stError {
        background-color: rgba(193, 123, 108, 0.1);
        border-left: 4px solid var(--bamboo-accent);
        color: var(--bamboo-text);
    }

    /* Info boxes */
    .stInfo {
        background-color: rgba(124, 148, 115, 0.1);
        border-left: 4px solid var(--bamboo-primary);
        color: var(--bamboo-text);
    }

    /* Captions */
    .caption {
        color: var(--bamboo-text);
        opacity: 0.7;
    }

    /* Code blocks */
    .stCodeBlock {
        background-color: #F5F7F2;
        border: 1px solid var(--bamboo-neutral);
    }

    code {
        font-size: 1.2rem !important;
    }

    /* Expander headers */
    .streamlit-expanderHeader {
        font-size: 1.4rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL & DATA
# ============================================================
from model import MNIST_CNN

@st.cache_resource
def load_model():
    model = MNIST_CNN()
    model.load_state_dict(torch.load('my_mnist_brain_CNN.pt', map_location='cpu'))
    model.eval()
    return model

@st.cache_data
def load_data():
    return datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

model = load_model()
test_data = load_data()

# ============================================================
# CALCULATE ACCURACY
# ============================================================
@st.cache_data
def calculate_accuracy():
    total = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_data:
            image_input = image.unsqueeze(0)
            output = model(image_input)
            predicted = torch.argmax(output, dim=1).item()
            if predicted == label:
                correct += 1
            total += 1

    return correct, total

correct, total = calculate_accuracy()
accuracy = (correct / total) * 100

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<h1 style='text-align: center; font-size: 3.5rem; color: #5F7C57;'>
    Can a Neural Network Read Your Handwriting?
</h1>
<p style='text-align: center; font-size: 1.6rem; color: #7C9473;'>
    A CNN trained on 60,000 digits - 99% accuracy
</p>
""", unsafe_allow_html=True)
st.divider()

# ============================================================
# METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{accuracy:.2f}%")
with col2:
    st.metric("Correct", f"{correct:,}")
with col3:
    st.metric("Total", f"{total:,}")
with col4:
    st.metric("Errors", f"{total - correct}")

st.divider()

# ============================================================
# PREDICTION EXPLORER
# ============================================================

index = st.slider("Test Image Index", 0, len(test_data)-1, 42, label_visibility="collapsed")

input_image, actual_label = test_data[index]
image_input = input_image.unsqueeze(0)

with torch.no_grad():
    output = model(image_input)
    probabilities = torch.softmax(output, dim=1)[0]
    predicted_label = torch.argmax(output, dim=1).item()
    confidence = probabilities[predicted_label].item()

# Symmetrical Layout
col1, col2 = st.columns([1, 1])

with col1:
    # Center the image
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)

    # Create figure with bamboo styling
    fig_img, ax_img = plt.subplots(figsize=(5, 5), facecolor='#F5F7F2')
    ax_img.imshow(input_image.squeeze(), cmap='gray')
    ax_img.axis('off')
    ax_img.set_facecolor('#F5F7F2')
    st.pyplot(fig_img)
    plt.close()

    st.markdown("</div>", unsafe_allow_html=True)

    # Clean status - centered
    if predicted_label == actual_label:
        st.success(f"**{predicted_label}** · {confidence:.1%} confidence", icon="✅")
    else:
        st.error(f"Predicted **{predicted_label}** · True **{actual_label}**", icon="❌")

    st.caption(f"Image #{index} from test set")

with col2:
    st.markdown("**Confidence Distribution**")
    st.caption("🟢 Highest | 🟠 2nd Highest | 🔵 3rd Highest")

    # Horizontal probability bars with color-coded top 3
    fig_bar, ax_bar = plt.subplots(figsize=(6, 5), facecolor='#F5F7F2')

    # Find top 3 predictions
    top3_indices = probabilities.argsort(descending=True)[:3].numpy()

    # Assign colors: Top 1 = Green, Top 2 = Orange, Top 3 = Blue, Rest = Gray
    colors = []
    for i in range(10):
        if i == top3_indices[0]:  # Highest (predicted)
            colors.append('#5F7C57')  # Green
        elif i == top3_indices[1]:  # 2nd highest
            colors.append('#E67E22')  # Orange
        elif i == top3_indices[2]:  # 3rd highest
            colors.append('#3498DB')  # Blue
        else:
            colors.append('#E8EBE4')  # Gray

    bars = ax_bar.barh(range(10), probabilities.numpy(), color=colors, height=0.7)

    ax_bar.set_yticks(range(10))
    ax_bar.set_yticklabels([str(i) for i in range(10)], fontsize=16, color='#3E4A3A')
    ax_bar.set_xlabel("Confidence", fontsize=14, color='#3E4A3A')
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_facecolor('#F5F7F2')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_color('#E8EBE4')
    ax_bar.spines['bottom'].set_color('#E8EBE4')
    ax_bar.grid(axis='x', alpha=0.3, color='#A8B899')
    ax_bar.tick_params(colors='#3E4A3A')

    # Add percentage labels on bars (show all top 3 + any others >2%)
    for i, (bar, prob) in enumerate(zip(bars, probabilities.numpy())):
        if i in top3_indices or prob > 0.02:
            ax_bar.text(prob + 0.01, i, f'{prob:.0%}',
                       va='center', fontsize=12, color='#3E4A3A', fontweight='bold' if i in top3_indices else 'normal')

    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close()

st.divider()

# ============================================================
# MODEL BEHAVIOR
# ============================================================

tab1, tab2, tab3 = st.tabs(["High Confidence ✅", "Uncertain 🤔", "Errors ❌"])

@st.cache_data
def find_cases(confidence_threshold=None, correct=None, max_search=2000, limit=6):
    """Generic function to find interesting cases"""
    cases = []
    for idx in range(max_search):
        if len(cases) >= limit:
            break
        img, label = test_data[idx]
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(output).item()
            conf = probs[pred].item()

            # Filter based on criteria
            if confidence_threshold is not None:
                if confidence_threshold == "high" and conf < 0.995:
                    continue
                if confidence_threshold == "low" and conf > 0.70:
                    continue

            if correct is not None:
                if correct and pred != label:
                    continue
                if not correct and pred == label:
                    continue

            cases.append((idx, img, label, pred, conf, probs))

    return cases

def display_case_grid(cases):
    """Display cases in a clean grid with bamboo styling"""
    cols = st.columns(6)
    for i, (idx, img, label, pred, conf, probs) in enumerate(cases):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(1.5, 1.5), facecolor='#F5F7F2')
            ax.imshow(img.squeeze(), cmap='gray')
            ax.axis('off')
            ax.set_facecolor('#F5F7F2')
            st.pyplot(fig)
            plt.close()

            if pred == label:
                st.markdown(f"<p style='color: #5F7C57; margin: 0; font-weight: 600;'>{pred}</p>", unsafe_allow_html=True)
                st.caption(f"{conf:.0%}")
            else:
                st.markdown(f"<p style='color: #C17B6C; margin: 0; font-weight: 600;'>~~{label}~~ → {pred}</p>", unsafe_allow_html=True)
                st.caption(f"{conf:.0%}")

with tab1:
    st.markdown("Examples where the model is **very confident** (>99.5%) and **correct**")
    cases = find_cases(confidence_threshold="high", correct=True, limit=6)
    if cases:
        display_case_grid(cases)
    else:
        st.info("No examples found with these criteria")

with tab2:
    st.markdown("Examples where the model is **uncertain** (<70% confidence)")
    cases = find_cases(confidence_threshold="low", limit=6)
    if cases:
        display_case_grid(cases)
    else:
        st.info("No examples found with these criteria")

with tab3:
    st.markdown("Examples where the model made **mistakes**")
    cases = find_cases(correct=False, limit=6)
    if cases:
        display_case_grid(cases)
    else:
        st.info("No errors found in the search!")

st.divider()

# ============================================================
# CODE VIEW
# ============================================================

with st.expander("📄 model.py - CNN Architecture"):
    st.code('''import torch.nn as nn

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

with st.expander("📄 app.py - Inference Script"):
    st.code('''import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import MNIST_CNN

# 1. Initialize and load the model
model = MNIST_CNN()
model.load_state_dict(torch.load('my_mnist_brain_CNN.pt',
                                 map_location='cpu'))
model.eval()

print("✅ Model loaded successfully!")

# Load test data
test_data = datasets.MNIST(root='./data', train=False,
                           download=True,
                           transform=transforms.ToTensor())

# 2. Calculate accuracy on full test set
print("\\n--- Testing on full test set ---")
total = 0
correct = 0

with torch.no_grad():
    for image, label in test_data:
        image_input = image.unsqueeze(0)
        output = model(image_input)
        predicted = torch.argmax(output, dim=1).item()

        if predicted == label:
            correct += 1
        total += 1

accuracy = (correct / total) * 100
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

# 3. Visualize a single prediction
index = 120
input_image, actual_label = test_data[index]
image_input = input_image.unsqueeze(0)

with torch.no_grad():
    output = model(image_input)
    predicted_label = torch.argmax(output, dim=1).item()

print(f"Actual: {actual_label}")
print(f"Predicted: {predicted_label}")

plt.imshow(input_image.squeeze(), cmap='gray')
plt.title(f"Actual: {actual_label} | Predicted: {predicted_label}")
plt.axis('off')
plt.show()
''', language='python')

# ============================================================
# DOWNLOAD SECTION
# ============================================================
st.divider()

try:
    with open('my_mnist_brain_CNN.pt', 'rb') as f:
        model_bytes = f.read()

    st.download_button(
        label="Download my_mnist_brain_CNN.pt",
        data=model_bytes,
        file_name="my_mnist_brain_CNN.pt",
        mime="application/octet-stream",
        help="Download the trained CNN model weights (PyTorch state_dict)"
    )
    st.caption("Use with: `model.load_state_dict(torch.load('my_mnist_brain_CNN.pt'))`")
except FileNotFoundError:
    st.warning("Model file not found. Make sure `my_mnist_brain_CNN.pt` exists in the project directory.")

# ============================================================
# DRAWING SECTION - TRY IT YOURSELF
# ============================================================
st.divider()

st.markdown("""
<div style='text-align: center; margin-bottom: 1rem;'>
    <p style='font-size: 1.5rem; color: #5F7C57;'>Draw a digit (0-9) and watch the model predict it instantly!</p>
</div>
""", unsafe_allow_html=True)

# Initialize canvas key in session state for clearing
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

col_draw1, col_draw2, col_draw3 = st.columns([1, 2, 1])

with col_draw2:
    draw_col1, draw_col2 = st.columns([1, 1])

    with draw_col1:
        # Drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )

        if st.button("Clear Canvas", use_container_width=True):
            st.session_state.canvas_key += 1
            st.rerun()

    with draw_col2:
        # Process and predict
        if canvas_result.image_data is not None:
            # Check if there's any drawing (not just black canvas)
            img_array = canvas_result.image_data

            # Convert to grayscale and check if there's content
            gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

            if np.max(gray) > 10:  # There's something drawn
                # MNIST-STYLE PREPROCESSING FOR HIGH ACCURACY
                coords = cv2.findNonZero(gray)
                x, y, w, h = cv2.boundingRect(coords)

                padding = 30
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(gray.shape[1], x + w + padding)
                y2 = min(gray.shape[0], y + h + padding)
                cropped = gray[y1:y2, x1:x2]

                h_crop, w_crop = cropped.shape
                if h_crop > w_crop:
                    diff = h_crop - w_crop
                    left_pad = diff // 2
                    right_pad = diff - left_pad
                    cropped = cv2.copyMakeBorder(cropped, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
                elif w_crop > h_crop:
                    diff = w_crop - h_crop
                    top_pad = diff // 2
                    bottom_pad = diff - top_pad
                    cropped = cv2.copyMakeBorder(cropped, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

                resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)
                img_28 = np.zeros((28, 28), dtype=np.uint8)
                img_28[4:24, 4:24] = resized

                M = cv2.moments(img_28)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shift_x = 14 - cx
                    shift_y = 14 - cy
                    M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    img_28 = cv2.warpAffine(img_28, M_shift, (28, 28))

                img_normalized = img_28.astype(np.float32) / 255.0
                img_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]
                    predicted_digit = torch.argmax(output, dim=1).item()
                    confidence = probabilities[predicted_digit].item()

                # Display prediction
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background: rgba(95, 124, 87, 0.1); border-radius: 10px;'>
                    <p style='font-size: 1.2rem; color: #7C9473; margin-bottom: 0.5rem;'>Prediction</p>
                    <p style='font-size: 6rem; font-weight: bold; color: #5F7C57; margin: 0; line-height: 1;'>{predicted_digit}</p>
                    <p style='font-size: 1.5rem; color: #7C9473;'>{confidence:.1%} confident</p>
                </div>
                """, unsafe_allow_html=True)

                # Show probability distribution
                st.markdown("<p style='text-align: center; margin-top: 1rem;'><b>All Probabilities:</b></p>", unsafe_allow_html=True)

                for digit in range(10):
                    prob = probabilities[digit].item()
                    bar_color = "#5F7C57" if digit == predicted_digit else "#E8EBE4"
                    st.markdown(f"""
                    <div style='display: flex; align-items: center; margin: 2px 0;'>
                        <span style='width: 20px; font-weight: bold;'>{digit}</span>
                        <div style='flex-grow: 1; background: #f0f0f0; height: 20px; border-radius: 3px; overflow: hidden;'>
                            <div style='width: {prob*100}%; background: {bar_color}; height: 100%;'></div>
                        </div>
                        <span style='width: 50px; text-align: right; font-size: 0.9rem;'>{prob:.0%}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align: center; padding: 3rem; background: rgba(95, 124, 87, 0.05); border-radius: 10px;'>
                    <p style='font-size: 1.3rem; color: #7C9473;'>Draw a digit on the canvas to see the prediction</p>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# QR CODE SECTION
# ============================================================
st.divider()

# Generate QR code for the app URL
# UPDATE THIS URL to your deployed Streamlit app URL
APP_URL = "https://mnist-push-g8ufuhpnd8bfjcnpqtdnnz.streamlit.app"

@st.cache_data
def generate_qr_code(url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#5F7C57", back_color="white")

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.markdown("""
    <div style='text-align: center;'>
        <h3 style='color: #5F7C57;'>Scan & Play!</h3>
        <p style='font-size: 1.2rem;'>Scan the QR code and try the drawing game on your phone</p>
    </div>
    """, unsafe_allow_html=True)

    qr_buffer = generate_qr_code(APP_URL)
    st.image(qr_buffer, width=250, use_container_width=False)

    st.markdown(f"""
    <p style='text-align: center; font-size: 1rem; color: #7C9473;'>
        Or visit: <a href="{APP_URL}" target="_blank" style="color: #5F7C57;">{APP_URL}</a>
    </p>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
