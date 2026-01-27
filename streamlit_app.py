import streamlit as st

st.set_page_config(page_title="Welcome to my App.py-", layout="wide")

st.markdown("""
<h1 style='text-align: center; font-size: 18rem; color: #5F7C57;'>
    Welcome to my App.py-
</h1>
""", unsafe_allow_html=True)

st.write("App is loading...")

# Test imports one by one
try:
    import torch
    st.success("torch loaded")
except Exception as e:
    st.error(f"torch failed: {e}")

try:
    from torchvision import datasets, transforms
    st.success("torchvision loaded")
except Exception as e:
    st.error(f"torchvision failed: {e}")

try:
    from model import MNIST_CNN
    st.success("model.py loaded")
except Exception as e:
    st.error(f"model.py failed: {e}")

try:
    import cv2
    st.success("cv2 loaded")
except Exception as e:
    st.error(f"cv2 failed: {e}")

try:
    from streamlit_drawable_canvas import st_canvas
    st.success("streamlit_drawable_canvas loaded")
except Exception as e:
    st.error(f"streamlit_drawable_canvas failed: {e}")

try:
    model = MNIST_CNN()
    model.load_state_dict(torch.load('my_mnist_brain_CNN.pt', map_location='cpu', weights_only=True))
    model.eval()
    st.success("Model weights loaded!")
except Exception as e:
    st.error(f"Model loading failed: {e}")

st.write("Done loading!")
