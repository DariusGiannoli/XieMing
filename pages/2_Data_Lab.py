import streamlit as st
import cv2
import numpy as np

# Set page configuration to match your app's theme
st.set_page_config(page_title="Data Lab", layout="wide")

st.title("🦅 Data Lab: Stereo Preparation")
st.write("Upload your stereo pair and configure augmentations to create a training batch.")

# 1. Initialize session state for batch storage
# This ensures the batch persists as you navigate between pages
if 'augmented_batch' not in st.session_state:
    st.session_state['augmented_batch'] = []
if 'stereo_pair' not in st.session_state:
    st.session_state['stereo_pair'] = {"left": None, "right": None}

# 2. Sidebar Parameters for Augmentation
# Users can interactively set the "stress test" limits for the models
st.sidebar.header("Augmentation Settings")
noise_intensity = st.sidebar.slider("Max Noise Intensity", 0.0, 1.0, 0.1)
brightness_range = st.sidebar.slider("Brightness Range", 0.5, 2.0, (0.8, 1.2))
rotation_range = st.sidebar.slider("Rotation Range (deg)", -45, 45, (-5, 5))
batch_size = st.sidebar.number_input("Batch Size to Generate", min_value=1, max_value=100, value=20)

def apply_augmentations(img, noise, bright, rot):
    """Applies CV2-based augmentations to a single image."""
    # Rotation
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), rot, 1.0)
    img = cv2.warpAffine(img, M, (w, h))
    
    # Brightness adjustment
    img = cv2.convertScaleAbs(img, alpha=bright, beta=0)
    
    # Add Gaussian Noise
    if noise > 0:
        gauss = np.random.normal(0, noise * 255, img.shape).astype('uint8')
        img = cv2.add(img, gauss)
    
    return img

# 3. Stereo Image Uploaders
col1, col2 = st.columns(2)
up_l = col1.file_uploader("Upload Left Image", type=['jpg', 'png'])
up_r = col2.file_uploader("Upload Right Image", type=['jpg', 'png'])

if up_l and up_r:
    # Read Images into OpenCV format
    l_bytes = np.asarray(bytearray(up_l.read()), dtype=np.uint8)
    img_l = cv2.imdecode(l_bytes, 1)
    
    r_bytes = np.asarray(bytearray(up_r.read()), dtype=np.uint8)
    img_r = cv2.imdecode(r_bytes, 1)
    
    # Store originals in session state
    st.session_state['stereo_pair'] = {"left": img_l, "right": img_r}
    
    # Display Original Pair
    st.subheader("Original Stereo Pair")
    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB), caption="Left Channel")
    c2.image(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB), caption="Right Channel")
    
    # 4. Interactive Live Preview
    st.subheader("Augmentation Preview")
    st.info("Adjust the sliders in the sidebar to see real-time effects on your data.")
    
    preview_l = apply_augmentations(img_l.copy(), noise_intensity, brightness_range[1], rotation_range[1])
    preview_r = apply_augmentations(img_r.copy(), noise_intensity, brightness_range[1], rotation_range[1])
    
    p1, p2 = st.columns(2)
    p1.image(cv2.cvtColor(preview_l, cv2.COLOR_BGR2RGB), caption="Augmented Left Preview")
    p2.image(cv2.cvtColor(preview_r, cv2.COLOR_BGR2RGB), caption="Augmented Right Preview")
    
    # 5. Batch Generation Button
    if st.button("🚀 Generate & Store Training Batch"):
        new_batch = []
        progress_bar = st.progress(0)
        
        for i in range(batch_size):
            # Randomly pick values within the user-defined slider ranges
            n = np.random.uniform(0, noise_intensity)
            b = np.random.uniform(brightness_range[0], brightness_range[1])
            r = np.random.uniform(rotation_range[0], rotation_range[1])
            
            aug_l = apply_augmentations(img_l.copy(), n, b, r)
            aug_r = apply_augmentations(img_r.copy(), n, b, r)
            
            # Store as pairs for stereo training
            new_batch.append({"left": aug_l, "right": aug_r})
            progress_bar.progress((i + 1) / batch_size)
        
        # Save to memory
        st.session_state['augmented_batch'] = new_batch
        st.success(f"Created {len(new_batch)} augmented samples. Ready for Model Tuning!")