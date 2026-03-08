import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detectors.rce.features import REGISTRY


# ---------------------------------------------------------------------------
# Cached model loaders — instantiated once, reused across reruns
# ---------------------------------------------------------------------------
@st.cache_resource
def load_resnet():
    from src.detectors.resnet import ResNetDetector
    return ResNetDetector()

@st.cache_resource
def load_mobilenet():
    from src.detectors.mobilenet import MobileNetDetector
    return MobileNetDetector()

@st.cache_resource
def load_mobilevit():
    from src.detectors.mobilevit import MobileViTDetector
    return MobileViTDetector()

st.set_page_config(page_title="Feature Lab", layout="wide")

if "pipeline_data" not in st.session_state:
    st.error("Please complete the Data Lab first!")
    st.stop()

assets = st.session_state["pipeline_data"]
# Use augmented crop if available, otherwise fall back to original crop
obj = assets.get("crop_aug", assets.get("crop"))
if obj is None:
    st.error("No crop found. Please go back to Data Lab and define a ROI.")
    st.stop()
gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)

st.title("🔬 Feature Lab: Physical Module Selection")

col_rce, col_cnn = st.columns([3, 2])

# ---------------------------------------------------------------------------
# LEFT — RCE Modular Engine (pure UI — all math lives in features.py)
# ---------------------------------------------------------------------------
with col_rce:
    st.header("🧬 RCE: Modular Physics Engine")
    st.subheader("Select Active Modules")

    # Dynamically build checkboxes from the registry
    m_cols = st.columns(len(REGISTRY))
    active = {}
    for i, (key, meta) in enumerate(REGISTRY.items()):
        active[key] = m_cols[i].checkbox(meta["label"], value=True)

    # Build vector + collect visualizations by calling registry functions
    final_vector = []
    viz_images = []
    for key, meta in REGISTRY.items():
        if active[key]:
            vec, viz = meta["fn"](gray)
            final_vector.extend(vec)
            viz_images.append((meta["viz_title"], viz))

    # Visualizations
    st.divider()
    if viz_images:
        v_cols = st.columns(len(viz_images))
        for idx, (title, img) in enumerate(viz_images):
            v_cols[idx].image(img, caption=title, use_container_width=True)
    else:
        st.warning("No modules selected — vector is empty.")

    # DNA vector bar chart
    st.write(f"### Current DNA Vector Size: **{len(final_vector)}**")
    fig_vec = go.Figure(data=[go.Bar(y=final_vector, marker_color="#00d4ff")])
    fig_vec.update_layout(title="Active Feature Vector (RCE Input)",
                          template="plotly_dark", height=300)
    st.plotly_chart(fig_vec, use_container_width=True)

# ---------------------------------------------------------------------------
# RIGHT — CNN comparison panel
# ---------------------------------------------------------------------------
with col_cnn:
    st.header("🧠 CNN: Static Architecture")
    selected_cnn = st.selectbox("Compare against Model", ["ResNet-18", "MobileViT-XXS", "MobileNetV3"])
    st.info("CNN features are fixed by pre-trained weights. You cannot toggle them like the RCE.")

    with st.spinner(f"Loading {selected_cnn} and extracting activations..."):
        try:
            if selected_cnn == "ResNet-18":
                detector = load_resnet()
                layer_name = "layer4 (last conv block)"
            elif selected_cnn == "MobileViT-XXS":
                detector = load_mobilevit()
                layer_name = "stages[-1] (last transformer stage)"
            else:
                detector = load_mobilenet()
                layer_name = "features[-1] (last features block)"

            act_maps = detector.get_activation_maps(obj, n_maps=6)
            st.caption(f"Hooked layer: `{layer_name}` — showing 6 of {len(act_maps)} channels")
            act_cols = st.columns(3)
            for i, amap in enumerate(act_maps):
                act_cols[i % 3].image(amap, caption=f"Channel {i}", use_container_width=True)

        except Exception as e:
            st.error(f"Could not load model: {e}")

    st.divider()
    st.markdown(f"""
    **Analysis:**
    - **Modularity:** RCE is **High** | CNN is **Zero**
    - **Explainability:** RCE is **High** | CNN is **Low**
    - **Compute Cost:** {len(final_vector)} floats | 512+ floats
    """)

# ---------------------------------------------------------------------------
# Lock configuration
# ---------------------------------------------------------------------------
if st.button("🚀 Lock Modular Configuration"):
    if not final_vector:
        st.error("Please select at least one module!")
    else:
        st.session_state["pipeline_data"]["final_vector"] = np.array(final_vector)
        st.session_state["active_modules"] = {k: v for k, v in active.items()}
        st.success("Modular DNA Locked! Ready for Model Tuning.")