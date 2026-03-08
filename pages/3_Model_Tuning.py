import streamlit as st
import cv2
import numpy as np
from src.detectors.yolo import YOLODetector
# Import your other detectors here

st.header("Recognition Benchmark")

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])

if uploaded_file:
    # Convert upload to OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Model Selection
    model = YOLODetector() # You can add a selectbox here
    label, conf, speed = model.predict(image)
    
    # UI Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    with col2:
        st.metric("Detected", label)
        st.metric("Confidence", f"{conf:.2%}")
        st.metric("Latency", f"{speed:.1f}ms")