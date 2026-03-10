import streamlit as st

st.set_page_config(page_title="Perception Benchmark", layout="wide", page_icon="🦅")

# ===================================================================
#  Routing — Sidebar Navigation
# ===================================================================
PIPELINES = {
    "🏠 Home": None,
    "📐 Stereo + Depth": {
        "🧪 Data Lab":          "tabs.stereo.data_lab",
        "🔬 Feature Lab":       "tabs.stereo.feature_lab",
        "⚙️ Model Tuning":      "tabs.stereo.model_tuning",
        "🔍 Localization Lab":  "tabs.stereo.localization",
        "🎯 Real-Time Detection":"tabs.stereo.detection",
        "📈 Evaluation":        "tabs.stereo.evaluation",
        "📐 Stereo Geometry":   "tabs.stereo.stereo_depth",
    },
    "🌍 Generalisation": {
        "🧪 Data Lab":          "tabs.generalisation.data_lab",
        "🔬 Feature Lab":       "tabs.generalisation.feature_lab",
        "⚙️ Model Tuning":      "tabs.generalisation.model_tuning",
        "🔍 Localization Lab":  "tabs.generalisation.localization",
        "🎯 Real-Time Detection":"tabs.generalisation.detection",
        "📈 Evaluation":        "tabs.generalisation.evaluation",
    },
}
h
st.sidebar.title("🦅 Recognition BenchMark")
pipeline_choice = st.sidebar.radio("Pipeline", list(PIPELINES.keys()), key="nav_pipeline")

stage_module = None
if PIPELINES[pipeline_choice] is not None:
    stages_map = PIPELINES[pipeline_choice]
    stage_choice = st.sidebar.radio("Stage", list(stages_map.keys()), key="nav_stage")
    module_path = stages_map[stage_choice]
    # dynamic import
    import importlib
    stage_module = importlib.import_module(module_path)

# Session status widget (always visible in sidebar)
st.sidebar.divider()
st.sidebar.subheader("📋 Session Status")

for pipe_label, pipe_key in [("Stereo", "stereo_pipeline"), ("General", "gen_pipeline")]:
    pipe = st.session_state.get(pipe_key, {})
    checks = {
        "Data locked":   "train_image" in pipe,
        "Crop defined":  "crop" in pipe,
        "Modules set":   "active_modules" in pipe,
        "RCE trained":   "rce_head" in pipe,
        "CNN trained":   any(f"cnn_head_{n}" in pipe
                             for n in ["ResNet-18", "MobileNetV3", "MobileViT-XXS"]),
        "Dets ready":    "rce_dets" in pipe or "cnn_dets" in pipe,
    }
    with st.sidebar.expander(f"**{pipe_label}** — {sum(checks.values())}/{len(checks)}"):
        for label, done in checks.items():
            st.markdown(f"{'✅' if done else '⬜'} {label}")

# ===================================================================
#  Home Page
# ===================================================================
def render_home():
    st.title("🦅 Recognition BenchMark")
    st.subheader("A stereo-vision pipeline for object recognition & depth estimation")
    st.caption("Compare classical feature engineering (RCE) against modern deep learning backbones — end-to-end, in your browser.")

    st.divider()

    # -------------------------------------------------------------------
    #  Two Pipelines
    # -------------------------------------------------------------------
    st.header("🗺️ Two Pipelines")
    st.markdown("""
    Choose a pipeline from the **sidebar**:

    - **📐 Stereo + Depth** — 7 stages. Uses a stereo image pair (LEFT=train, RIGHT=test)
      with calibration data and ground-truth disparities. Ends with metric depth estimation.
    - **🌍 Generalisation** — 6 stages. Uses different scene *variants* from the Middlebury dataset
      (train on one variant, test on another). Tests how well models generalise across viewpoints.
    """)

    col_s, col_g = st.columns(2)
    with col_s:
        st.markdown("### 📐 Stereo + Depth (7 stages)")
        stereo_stages = [
            ("🧪", "Data Lab",          "Load stereo pair, calib, GT depth. Define ROIs."),
            ("🔬", "Feature Lab",        "Toggle RCE modules, compare CNN activations."),
            ("⚙️", "Model Tuning",       "Train RCE / CNN / ORB heads."),
            ("🔍", "Localization Lab",   "Compare 5 localization strategies."),
            ("🎯", "Real-Time Detection","Sliding window on the TEST image."),
            ("📈", "Evaluation",         "Confusion matrices, PR curves, F1."),
            ("📐", "Stereo Geometry",    "StereoSGBM disparity → metric depth."),
        ]
        for icon, title, desc in stereo_stages:
            st.markdown(f"{icon} **{title}** — {desc}")

    with col_g:
        st.markdown("### 🌍 Generalisation (6 stages)")
        gen_stages = [
            ("🧪", "Data Lab",          "Pick scene group & variants (train ≠ test)."),
            ("🔬", "Feature Lab",        "Toggle RCE modules, compare CNN activations."),
            ("⚙️", "Model Tuning",       "Train RCE / CNN / ORB heads."),
            ("🔍", "Localization Lab",   "Compare 5 localization strategies."),
            ("🎯", "Real-Time Detection","Sliding window on a different variant."),
            ("📈", "Evaluation",         "Confusion matrices, PR curves, F1."),
        ]
        for icon, title, desc in gen_stages:
            st.markdown(f"{icon} **{title}** — {desc}")

    st.divider()

    # -------------------------------------------------------------------
    #  Models
    # -------------------------------------------------------------------
    st.header("🧠 Models Used")

    tab_rce, tab_resnet, tab_mobilenet, tab_mobilevit = st.tabs(
        ["RCE Engine", "ResNet-18", "MobileNetV3-Small", "MobileViT-XXS"])

    with tab_rce:
        st.markdown("### 🧬 RCE — Relative Contextual Encoding")
        st.markdown("""
**Type:** Modular hand-crafted feature extractor
**Architecture:** Seven physics-inspired modules, each producing a 10-bin histogram:

| Module | Input | Operation |
|--------|-------|-----------|
| **Intensity** | Grayscale | Pixel-value histogram (global appearance) |
| **Sobel** | Gradient magnitude | Edge strength distribution (texture) |
| **Spectral** | FFT log-magnitude | Frequency content (pattern / structure) |
| **Laplacian** | Laplacian response | Second-derivative focus / sharpness |
| **Gradient Orientation** | Sobel angles | Edge direction histogram |
| **Gabor** | Multi-kernel response | Texture at multiple orientations / scales |
| **LBP** | Local Binary Patterns | Micro-texture descriptor |

Max feature vector = **70D** (7 modules × 10 bins).
        """)

    with tab_resnet:
        st.markdown("### 🏗️ ResNet-18")
        st.markdown("""
**Source:** PyTorch Hub (`torchvision.models.ResNet18_Weights.DEFAULT`)
**Pre-training:** ImageNet-1k (1.28 M images, 1 000 classes)
**Backbone output:** 512-dimensional embedding (after `avgpool`)
**Head:** LogisticRegression trained on your session data

**In this app:** The entire backbone is **frozen** (`requires_grad=False`).
Only the lightweight head adapts to your specific object.
        """)

    with tab_mobilenet:
        st.markdown("### 📱 MobileNetV3-Small")
        st.markdown("""
**Source:** PyTorch Hub (`torchvision.models.MobileNet_V3_Small_Weights.DEFAULT`)
**Pre-training:** ImageNet-1k
**Backbone output:** 576-dimensional embedding
**Head:** LogisticRegression trained on your session data

**In this app:** Typically 3–5× faster than ResNet-18.
        """)

    with tab_mobilevit:
        st.markdown("### 🤖 MobileViT-XXS")
        st.markdown("""
**Source:** timm — `mobilevit_xxs.cvnets_in1k` (Apple Research, 2022)
**Pre-training:** ImageNet-1k
**Backbone output:** 320-dimensional embedding (`num_classes=0`)
**Head:** LogisticRegression trained on your session data

**In this app:** Hybrid CNN + Vision Transformer. Only ~1.3 M parameters.
        """)

    st.divider()

    # -------------------------------------------------------------------
    #  Depth Estimation
    # -------------------------------------------------------------------
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
        st.markdown("**Depth formula (Middlebury convention):**")
        st.latex(r"Z = \frac{f \times B}{d + d_{\text{offs}}}")
        st.markdown("""
- $f$ — focal length (pixels)
- $B$ — baseline (mm, from calibration file)
- $d$ — disparity (pixels)
- $d_\\text{offs}$ — optical-center offset between cameras
        """)

    st.divider()
    st.caption("Select a pipeline from the **sidebar** to begin.")


# ===================================================================
#  Dispatch
# ===================================================================
if stage_module is not None:
    stage_module.render()
else:
    render_home()