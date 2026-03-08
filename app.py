import streamlit as st

st.set_page_config(page_title="Perception Benchmark", layout="wide", page_icon="🦅")

# ===================================================================
#  Header
# ===================================================================
st.title("🦅 Recognition BenchMark")
st.subheader("A stereo-vision pipeline for object recognition & depth estimation")
st.caption("Compare classical feature engineering (RCE) against modern deep learning backbones — end-to-end, in your browser.")

st.divider()

# ===================================================================
#  Pipeline Overview
# ===================================================================
st.header("🗺️ Pipeline Overview")
st.markdown("""
The app is structured as a **5-stage sequential pipeline**.
Complete each page in order — every stage feeds the next.
""")

stages = [
    ("🧪", "1 · Data Lab",          "Upload a stereo image pair, camera calibration file, and two PFM ground-truth depth maps. "
                                      "Define an object ROI (bounding box), then apply live data augmentation "
                                      "(brightness, contrast, rotation, noise, blur, shift, flip). "
                                      "All assets are locked into session state — nothing is written to disk."),
    ("🔬", "2 · Feature Lab",        "Toggle RCE physics modules (Intensity · Sobel · Spectral) to build a modular "
                                      "feature vector. Compare it live against CNN activation maps extracted from a "
                                      "frozen backbone via forward hooks. Lock your active module configuration."),
    ("⚙️", "3 · Model Tuning",       "Train lightweight **heads** on your session data (augmented crop = positives, "
                                      "random non-overlapping patches from the scene = negatives). "
                                      "Both RCE and CNN heads are trained identically with LogisticRegression "
                                      "and stored in session state only — no disk writes."),
    ("🎯", "4 · Real-Time Detection","Run a **sliding window** across the right image using both the RCE head and "
                                      "your chosen CNN head simultaneously. Watch the scan live, then compare "
                                      "bounding boxes, confidence heatmaps, and latency."),
    ("📐", "5 · Stereo Geometry",    "Compute a disparity map with **StereoSGBM**, convert it to metric depth "
                                      "using the stereo formula $Z = fB/(d+d_{\\text{offs}})$, then read depth "
                                      "directly at every detected bounding box. Compare against PFM ground truth."),
]

for icon, title, desc in stages:
    with st.container(border=True):
        c1, c2 = st.columns([1, 12])
        c1.markdown(f"## {icon}")
        c2.markdown(f"**{title}**  \n{desc}")

st.divider()

# ===================================================================
#  Models
# ===================================================================
st.header("🧠 Models Used")

tab_rce, tab_resnet, tab_mobilenet, tab_mobilevit = st.tabs(
    ["RCE Engine", "ResNet-18", "MobileNetV3-Small", "MobileViT-XXS"])

with tab_rce:
    st.markdown("### 🧬 RCE — Relative Contextual Encoding")
    st.markdown("""
**Type:** Modular hand-crafted feature extractor  
**Architecture:** Three physics-inspired modules, each producing a 10-bin histogram:

| Module | Input | Operation |
|--------|-------|-----------|
| **Intensity** | Grayscale | Pixel-value histogram (global appearance) |
| **Sobel** | Gradient magnitude | Edge strength distribution (texture) |
| **Spectral** | FFT log-magnitude | Frequency content (pattern / structure) |

**Strengths:**
- Fully explainable — every dimension has a physical meaning
- Extremely fast (µs per patch, no GPU needed)
- Modular: disable any module and immediately see the effect on the vector
- Zero pre-training needed

**Weakness:** Less discriminative than deep features for complex visual scenes.
    """)

with tab_resnet:
    st.markdown("### 🏗️ ResNet-18")
    st.markdown("""
**Source:** PyTorch Hub (`torchvision.models.ResNet18_Weights.DEFAULT`)  
**Pre-training:** ImageNet-1k (1.28 M images, 1 000 classes)  
**Backbone output:** 512-dimensional embedding (after `avgpool`)  
**Head:** LogisticRegression trained on your session data

**Architecture highlights:**
- 18 layers with residual (skip) connections
- Residual blocks prevent vanishing gradients in deeper networks
- `layer4` is hooked for activation map visualisation

**In this app:** The entire backbone is **frozen** (`requires_grad=False`).  
Only the lightweight head adapts to your specific object.
    """)

with tab_mobilenet:
    st.markdown("### 📱 MobileNetV3-Small")
    st.markdown("""
**Source:** PyTorch Hub (`torchvision.models.MobileNet_V3_Small_Weights.DEFAULT`)  
**Pre-training:** ImageNet-1k  
**Backbone output:** 576-dimensional embedding (classifier replaced with `Identity`)  
**Head:** LogisticRegression trained on your session data

**Architecture highlights:**
- Inverted residuals + linear bottlenecks (MobileNetV2 heritage)
- Hard-Swish / Hard-Sigmoid activations (hardware-friendly)
- Squeeze-and-Excitation (SE) blocks for channel attention
- Designed for **edge / mobile inference** — ~2.5 M parameters

**In this app:** Typically 3–5× faster than ResNet-18.  
`features[-1]` is hooked for activation maps.
    """)

with tab_mobilevit:
    st.markdown("### 🤖 MobileViT-XXS")
    st.markdown("""
**Source:** timm — `mobilevit_xxs.cvnets_in1k` (Apple Research, 2022)  
**Pre-training:** ImageNet-1k  
**Backbone output:** 320-dimensional embedding (`num_classes=0`)  
**Head:** LogisticRegression trained on your session data

**Architecture highlights:**
- **Hybrid CNN + Vision Transformer** — local convolutions for spatial features,
  global self-attention for long-range context
- MobileNetV2 stem + MobileViT blocks (attention on non-overlapping patches)
- Only ~1.3 M parameters — smallest of the three

**In this app:** The final transformer stage `stages[-1]` is hooked.  
Slower than MobileNetV3 but captures global structure.
    """)

st.divider()

# ===================================================================
#  Depth Estimation
# ===================================================================
st.header("📐 Stereo Depth Estimation")

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.markdown("""
**Algorithm:** `cv2.StereoSGBM` (Semi-Global Block Matching)

SGBM minimises a global energy function combining:
- Data cost (pixel intensity difference)
- Smoothness penalty (P1, P2 regularisation)

It processes multiple horizontal and diagonal scan-line passes, 
making it significantly more accurate than basic block matching.
    """)
with col_d2:
    st.markdown("""
**Depth formula (Middlebury convention):**
    """)
    st.latex(r"Z = \frac{f \times B}{d + d_{\text{offs}}}")
    st.markdown("""
- $f$ — focal length (pixels)  
- $B$ — baseline (mm, from calibration file)  
- $d$ — disparity (pixels)  
- $d_\\text{offs}$ — optical-center offset between cameras
    """)

st.divider()

# ===================================================================
#  Session Status
# ===================================================================
st.header("📋 Session Status")

pipe = st.session_state.get("pipeline_data", {})

checks = {
    "Data Lab locked":       "left" in pipe,
    "Crop defined":          "crop" in pipe,
    "Augmentation applied":  "crop_aug" in pipe,
    "Active modules locked": "active_modules" in st.session_state,
    "RCE head trained":      "rce_head" in st.session_state,
    "CNN head trained":      any(f"cnn_head_{n}" in st.session_state
                                  for n in ["ResNet-18", "MobileNetV3", "MobileViT-XXS"]),
    "RCE detections ready":  "rce_dets" in st.session_state,
    "CNN detections ready":  "cnn_dets" in st.session_state,
}

cols = st.columns(4)
for i, (label, done) in enumerate(checks.items()):
    cols[i % 4].markdown(
        f"{'✅' if done else '⬜'} {'~~' if not done else ''}{label}{'~~' if not done else ''}"
    )

st.divider()
st.caption("Navigate using the sidebar → Start with **🧪 Data Lab**")