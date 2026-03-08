import streamlit as st
import cv2
import numpy as np
import time
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors.rce.features import REGISTRY
from src.models import BACKBONES, RecognitionHead

st.set_page_config(page_title="Model Tuning", layout="wide")
st.title("⚙️ Model Tuning: Train & Compare")

# ---------------------------------------------------------------------------
# Guard: require Data Lab completion
# ---------------------------------------------------------------------------
if "pipeline_data" not in st.session_state or "crop" not in st.session_state.get("pipeline_data", {}):
    st.error("Please complete the **Data Lab** first (upload assets & define a crop).")
    st.stop()

assets = st.session_state["pipeline_data"]
crop      = assets["crop"]           # original crop from Data Lab
crop_aug  = assets.get("crop_aug", crop)  # augmented crop from Data Lab
left_img  = assets["left"]           # full left image
bbox      = assets.get("crop_bbox", (0, 0, crop.shape[1], crop.shape[0]))
active_modules = st.session_state.get("active_modules", {k: True for k in REGISTRY})


# ---------------------------------------------------------------------------
# Build training set from session data (no disk reads)
# ---------------------------------------------------------------------------
def build_training_set(augment_fn=None):
    """
    Positive samples:  original crop + augmented crop from Data Lab.
    Negative samples:  random patches from the left image that do NOT
                       overlap with the crop bounding box.
    Returns (images_list, labels_list).
    """
    positives = [crop, crop_aug]
    if augment_fn is not None:
        positives.append(augment_fn(crop))

    # --- Generate negatives from left image margins ---
    x0, y0, x1, y1 = bbox
    H, W = left_img.shape[:2]
    ch, cw = y1 - y0, x1 - x0  # crop height/width
    negatives = []
    rng = np.random.default_rng(42)

    attempts = 0
    while len(negatives) < len(positives) * 2 and attempts < 200:
        # Random patch of same size as crop
        rx = rng.integers(0, max(W - cw, 1))
        ry = rng.integers(0, max(H - ch, 1))
        # Reject if it overlaps the crop bbox (IoU > 0)
        if rx < x1 and rx + cw > x0 and ry < y1 and ry + ch > y0:
            attempts += 1
            continue
        patch = left_img[ry:ry+ch, rx:rx+cw]
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            negatives.append(patch)
        attempts += 1

    images = positives + negatives
    labels = ["object"] * len(positives) + ["background"] * len(negatives)
    return images, labels


def build_rce_vector(img_bgr):
    """Build the RCE feature vector from active modules."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    vec = []
    for key, meta in REGISTRY.items():
        if active_modules.get(key, False):
            v, _ = meta["fn"](gray)
            vec.extend(v)
    return np.array(vec, dtype=np.float32)


# ===================================================================
# Show data used for training
# ===================================================================
st.subheader("Training Data (from Data Lab)")
st.caption("Positives = your crop + augmented crop  |  Negatives = random non-overlapping patches from left image")
td1, td2 = st.columns(2)
td1.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="Original Crop (positive)", width=180)
td2.image(cv2.cvtColor(crop_aug, cv2.COLOR_BGR2RGB), caption="Augmented Crop (positive)", width=180)

st.divider()

# ===================================================================
# LAYOUT: LEFT = RCE  |  RIGHT = CNN
# ===================================================================
col_rce, col_cnn = st.columns(2)

# ---------------------------------------------------------------------------
# LEFT — RCE Training
# ---------------------------------------------------------------------------
with col_rce:
    st.header("🧬 RCE Training")

    active_names = [REGISTRY[k]["label"] for k in active_modules if active_modules[k]]
    if not active_names:
        st.error("No RCE modules selected. Go back to Feature Lab.")
        st.stop()
    st.write(f"**Active modules:** {', '.join(active_names)}")

    st.subheader("Training Parameters")
    rce_C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, step=0.01,
                       help="Higher = less regularization, may overfit")
    rce_max_iter = st.slider("Max Iterations", 100, 5000, 1000, step=100)

    if st.button("🚀 Train RCE Head"):
        images, labels = build_training_set()
        from sklearn.metrics import accuracy_score

        progress = st.progress(0, text="Extracting RCE features...")
        n = len(images)
        X = []
        for i, img in enumerate(images):
            X.append(build_rce_vector(img))
            progress.progress((i + 1) / n, text=f"Feature extraction: {i+1}/{n}")

        X = np.array(X)
        progress.progress(1.0, text="Fitting Logistic Regression...")

        t0 = time.perf_counter()
        head = RecognitionHead(C=rce_C, max_iter=rce_max_iter).fit(X, labels)
        train_time = time.perf_counter() - t0
        progress.progress(1.0, text="✅ Training complete!")

        preds = head.model.predict(X)
        train_acc = accuracy_score(labels, preds)

        st.success(f"Trained in **{train_time:.2f}s**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Train Accuracy", f"{train_acc:.1%}")
        m2.metric("Vector Size", f"{X.shape[1]} floats")
        m3.metric("Samples", f"{len(images)}")

        probs = head.predict_proba(X)
        fig = go.Figure()
        for ci, cls in enumerate(head.classes_):
            fig.add_trace(go.Histogram(x=probs[:, ci], name=cls, opacity=0.7, nbinsx=20))
        fig.update_layout(title="Confidence Distribution", barmode="overlay",
                          template="plotly_dark", height=280,
                          xaxis_title="Confidence", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        # Store head in session (no disk save)
        st.session_state["rce_head"] = head
        st.session_state["rce_train_acc"] = train_acc

    if "rce_head" in st.session_state:
        st.divider()
        st.subheader("Quick Predict (Crop)")
        head = st.session_state["rce_head"]
        t0 = time.perf_counter()
        vec = build_rce_vector(crop_aug)
        label, conf = head.predict(vec)
        dt = (time.perf_counter() - t0) * 1000
        st.write(f"**{label}** — {conf:.1%} confidence — {dt:.1f} ms")


# ---------------------------------------------------------------------------
# RIGHT — CNN Fine-Tuning
# ---------------------------------------------------------------------------
with col_cnn:
    st.header("🧠 CNN Fine-Tuning")

    selected = st.selectbox("Select Model", list(BACKBONES.keys()))
    meta = BACKBONES[selected]
    st.caption(f"Backbone embedding: **{meta['dim']}D** → Logistic Regression head")

    st.subheader("Training Parameters")
    cnn_C = st.slider("Regularization (C) ", 0.01, 10.0, 1.0, step=0.01,
                       key="cnn_c", help="Higher = less regularization")
    cnn_max_iter = st.slider("Max Iterations ", 100, 5000, 1000, step=100,
                              key="cnn_iter")

    if st.button(f"🚀 Train {selected} Head"):
        images, labels = build_training_set()
        backbone = meta["loader"]()          # cached frozen backbone

        from sklearn.metrics import accuracy_score

        progress = st.progress(0, text=f"Extracting {selected} features...")
        n = len(images)
        X = []
        for i, img in enumerate(images):
            X.append(backbone.get_features(img))
            progress.progress((i + 1) / n, text=f"Feature extraction: {i+1}/{n}")

        X = np.array(X)
        progress.progress(1.0, text="Fitting Logistic Regression...")

        t0 = time.perf_counter()
        head = RecognitionHead(C=cnn_C, max_iter=cnn_max_iter).fit(X, labels)
        train_time = time.perf_counter() - t0
        progress.progress(1.0, text="✅ Training complete!")

        preds = head.model.predict(X)
        train_acc = accuracy_score(labels, preds)

        st.success(f"Trained in **{train_time:.2f}s**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Train Accuracy", f"{train_acc:.1%}")
        m2.metric("Vector Size", f"{X.shape[1]}D")
        m3.metric("Samples", f"{len(images)}")

        probs = head.predict_proba(X)
        fig = go.Figure()
        for ci, cls in enumerate(head.classes_):
            fig.add_trace(go.Histogram(x=probs[:, ci], name=cls, opacity=0.7, nbinsx=20))
        fig.update_layout(title="Confidence Distribution", barmode="overlay",
                          template="plotly_dark", height=280,
                          xaxis_title="Confidence", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        # Store head in session (no disk save)
        st.session_state[f"cnn_head_{selected}"] = head
        st.session_state[f"cnn_acc_{selected}"] = train_acc

    if f"cnn_head_{selected}" in st.session_state:
        st.divider()
        st.subheader("Quick Predict (Crop)")
        backbone = meta["loader"]()          # cached frozen backbone
        head = st.session_state[f"cnn_head_{selected}"]
        t0 = time.perf_counter()
        feats = backbone.get_features(crop_aug)
        label, conf = head.predict(feats)
        dt = (time.perf_counter() - t0) * 1000
        st.write(f"**{label}** — {conf:.1%} confidence — {dt:.1f} ms")


# ===========================================================================
# Bottom — Side-by-side comparison table
# ===========================================================================
st.divider()
st.subheader("📊 Training Comparison")

rce_acc = st.session_state.get("rce_train_acc")
rows = []
if rce_acc is not None:
    rows.append({"Model": "RCE", "Train Accuracy": f"{rce_acc:.1%}",
                 "Vector Size": str(sum(10 for k in active_modules if active_modules[k]))})
for name in BACKBONES:
    acc = st.session_state.get(f"cnn_acc_{name}")
    if acc is not None:
        rows.append({"Model": name, "Train Accuracy": f"{acc:.1%}",
                     "Vector Size": f"{BACKBONES[name]['dim']}D"})

if rows:
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("Train at least one model to see the comparison.")