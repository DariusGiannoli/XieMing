"""Stereo Feature Lab — Stage 2 of the Stereo + Depth pipeline."""

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go

from src.detectors.rce.features import REGISTRY
from src.models import BACKBONES


def render():
    pipe = st.session_state.get("stereo_pipeline")
    if not pipe or "crop" not in pipe:
        st.error("Please complete the **Data Lab** first!")
        st.stop()

    obj = pipe.get("crop_aug", pipe.get("crop"))
    if obj is None:
        st.error("No crop found. Go back to Data Lab and define a ROI.")
        st.stop()
    gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)

    st.title("🔬 Feature Lab: Physical Module Selection")

    col_rce, col_cnn = st.columns([3, 2])

    # -------------------------------------------------------------------
    # LEFT — RCE Modular Engine
    # -------------------------------------------------------------------
    with col_rce:
        st.header("🧬 RCE: Modular Physics Engine")
        st.subheader("Select Active Modules")

        active = {}
        items = list(REGISTRY.items())
        for row_start in range(0, len(items), 4):
            row_items = items[row_start:row_start + 4]
            m_cols = st.columns(4)
            for col, (key, meta) in zip(m_cols, row_items):
                active[key] = col.checkbox(meta["label"],
                    value=(key in ("intensity", "sobel", "spectral")),
                    key=f"stereo_fl_{key}")

        final_vector = []
        viz_images = []
        for key, meta in REGISTRY.items():
            if active[key]:
                vec, viz = meta["fn"](gray)
                final_vector.extend(vec)
                viz_images.append((meta["viz_title"], viz))

        st.divider()
        if viz_images:
            for row_start in range(0, len(viz_images), 3):
                row = viz_images[row_start:row_start + 3]
                v_cols = st.columns(3)
                for col, (title, img) in zip(v_cols, row):
                    col.image(img, caption=title, use_container_width=True)
        else:
            st.warning("No modules selected — vector is empty.")

        st.write(f"### Current DNA Vector Size: **{len(final_vector)}**")
        fig_vec = go.Figure(data=[go.Bar(y=final_vector, marker_color="#00d4ff")])
        fig_vec.update_layout(title="Active Feature Vector (RCE Input)",
                              template="plotly_dark", height=300)
        st.plotly_chart(fig_vec, use_container_width=True)

    # -------------------------------------------------------------------
    # RIGHT — CNN comparison panel
    # -------------------------------------------------------------------
    with col_cnn:
        st.header("🧠 CNN: Static Architecture")
        selected_cnn = st.selectbox("Compare against Model", list(BACKBONES.keys()),
                                    key="stereo_fl_cnn")
        st.info("CNN features are fixed by pre-trained weights. "
                "You cannot toggle them like the RCE.")

        with st.spinner(f"Loading {selected_cnn} and extracting activations..."):
            try:
                bmeta = BACKBONES[selected_cnn]
                backbone = bmeta["loader"]()
                layer_name = bmeta["hook_layer"]

                act_maps = backbone.get_activation_maps(obj, n_maps=6)
                st.caption(f"Hooked layer: `{layer_name}` — showing 6 of "
                           f"{len(act_maps)} channels")
                act_cols = st.columns(3)
                for i, amap in enumerate(act_maps):
                    act_cols[i % 3].image(amap, caption=f"Channel {i}",
                                          use_container_width=True)
            except Exception as e:
                st.error(f"Could not load model: {e}")

        st.divider()
        st.markdown(f"""
        **Analysis:**
        - **Modularity:** RCE is **High** | CNN is **Zero**
        - **Explainability:** RCE is **High** | CNN is **Low**
        - **Compute Cost:** {len(final_vector)} floats | 512+ floats
        """)

    # -------------------------------------------------------------------
    # Lock configuration
    # -------------------------------------------------------------------
    if st.button("🚀 Lock Modular Configuration", key="stereo_fl_lock"):
        if not final_vector:
            st.error("Please select at least one module!")
        else:
            pipe["final_vector"] = np.array(final_vector)
            pipe["active_modules"] = {k: v for k, v in active.items()}
            st.session_state["stereo_pipeline"] = pipe
            st.success("Modular DNA Locked! Ready for Model Tuning.")
