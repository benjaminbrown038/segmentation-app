# app.py
import streamlit as st
from segment import load_model, segment_image
from PIL import Image

st.title("🧩 Semantic Segmentation App")
st.write("Upload an image and visualize pixel-level segmentation using DeepLabV3 (ResNet-34 backbone).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Running segmentation...")

    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    model = load_model()
    overlay, mask = segment_image(model, "temp.jpg")

    st.subheader("Segmentation Output:")
    st.image(overlay, caption="Overlay Result", use_container_width=True)
