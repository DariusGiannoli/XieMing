import streamlit as st
import cv2
import numpy as np
import time
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors.rce.features import REGISTRY
from src.models import BACKBONES, RecognitionHead
from src.utils import build_rce_vector

st.set_page_config(page_title="Model Tuning", layout="wide")
st.title("⚙️ Model Tuning: Train & Compare")

# ---------------------------------------------------------------------------
# Guard: require Data Lab completion
# ---------------------------------------------------------------------------
if "pipeline_data" not in st.session_state or "crop" not in st.session_state.get("pipeline_data", {}):
    st.error("Please complete the **Data Lab** first (upload assets & define a crop).")
    st.stop()

assets = st.session_state["pipeline_data"]
crop      = assets["crop"]
crop_aug  = assets.get("crop_aug", crop)
left_img  = assets["left"]
bbox      = assets.get("crop_bbox", (0, 0, crop.shape[1], crop.shape[0]))
rois      = assets.get("rois", [{"label": "object", "bbox": bbox,
                                  "crop": crop, "crop_aug": crop_aug}])
active_modules = st.session_state.get("active_modules", {k: True for k in REGISTRY})

is_multi = len(rois) > 1

# ---------------------------------------------------------------------------
# Build training set from session data (no disk reads)
# ---------------------------------------------------------------------------
def build_training_set():
    """
    Multi-class aware training set builder.
    Positive samples per class: original crop + augmented crop.
    Negative samples: random patches that don't overlap ANY ROI.
    """
    images = []
    labels = []

    for roi in rois:
        images.append(roi["crop"])
        labels.append(roi["label"])
        images.append(roi["crop_aug"])
        labels.append(roi["label"])

    all_bboxes = [roi["bbox"] for roi in rois]
    H, W = left_img.shape[:2]
    x0r, y0r, x1r, y1r = rois[0]["bbox"]
    ch, cw = y1r - y0r, x1r - x0r
    rng = np.random.default_rng(42)

    n_neg_target = len(images) * 2
    attempts = 0
    negatives = []
    while len(negatives) < n_neg_target and attempts < 300:
        rx = rng.integers(0, max(W - cw, 1))
        ry = rng.integers(0, max(H - ch, 1))
        overlaps = False
        for bx0, by0, bx1, by1 in all_bboxes:
            if rx < bx1 and rx + cw > bx0 and ry < by1 and ry + ch > by0:
                overlaps = True
                break
        if overlaps:
            attempts += 1
            continue
        patch = left_img[ry:ry+ch, rx:rx+cw]
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            negatives.append(patch)
        attempts += 1

    images.extend(negatives)
    labels.extend(["background"] * len(negatives))
    return images, labels, len(negatives) < n_neg_target // 2


# ===================================================================
# Show training data
# ===================================================================
st.subheader("Training Data (from Data Lab)")
if is_multi:
    st.caption(f"**{len(rois)} classes** defined — each ROI becomes a separate class.")
    roi_cols = st.columns(min(len(rois), 4))
    for i, roi in enumerate(rois):
        with roi_cols[i % len(roi_cols)]:
            st.image(cv2.cvtColor(roi["crop"], cv2.COLOR_BGR2RGB),
                     caption=f"✅ {roi['label']}", width=140)
else:
    st.caption("Positives = your crop + augmented crop  |  "
               "Negatives = random non-overlapping patches")
    td1, td2 = st.columns(2)
    td1.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
              caption="Original Crop (positive)", width=180)
    td2.image(cv2.cvtColor(crop_aug, cv2.COLOR_BGR2RGB),
              caption="Augmented Crop (positive)", width=180)

st.divider()

# ===================================================================
# LAYOUT: RCE  |  CNN  |  ORB
# ===================================================================
col_rce, col_cnn, col_orb = st.columns(3)

# ---------------------------------------------------------------------------
# LEFT — RCE Training
# ---------------------------------------------------------------------------
with col_rce:
    st.header("🧬 RCE Training")

    active_names = [REGISTRY[k]["label"] for k in active_modules if active_modules[k]]
    if not active_names:
        st.error("No RCE modules selected. Go back to Feature Lab.")
    else:
        st.write(f"**Active modules:** {', '.join(active_names)}")

        st.subheader("Training Parameters")
        rce_C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, step=0.01,
                           help="Higher = less regularization, may overfit")
        rce_max_iter = st.slider("Max Iterations", 100, 5000, 1000, step=100)

        if st.button("🚀 Train RCE Head"):
            images, labels, neg_short = build_training_set()
            if neg_short:
                st.warning(f"⚠️ Only {sum(1 for l in labels if l == 'background')} "
                           f"negatives collected (target was {sum(1 for l in labels if l != 'background') * 2}). "
                           f"Training data may be imbalanced.")
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score

            progress = st.progress(0, text="Extracting RCE features...")
            n = len(images)
            X = []
            for i, img in enumerate(images):
                X.append(build_rce_vector(img, active_modules))
                progress.progress((i + 1) / n, text=f"Feature extraction: {i+1}/{n}")

            X = np.array(X)
            progress.progress(1.0, text="Fitting Logistic Regression...")

            t0 = time.perf_counter()
            try:
                head = RecognitionHead(C=rce_C, max_iter=rce_max_iter).fit(X, labels)
            except ValueError as e:
                st.error(f"Training failed: {e}")
                st.stop()
            train_time = time.perf_counter() - t0
            progress.progress(1.0, text="✅ Training complete!")

            preds = head.model.predict(X)
            train_acc = accuracy_score(labels, preds)

            st.success(f"Trained in **{train_time:.2f}s**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train Accuracy", f"{train_acc:.1%}")
            # Cross-validation (only if enough samples)
            if len(images) >= 6:
                n_splits = min(5, len(set(labels)))
                if n_splits >= 2:
                    cv_scores = cross_val_score(head.model, X, labels,
                                               cv=min(3, len(images) // 2))
                    m2.metric("CV Accuracy", f"{cv_scores.mean():.1%}",
                              delta=f"±{cv_scores.std():.1%}")
                else:
                    m2.metric("CV Accuracy", "N/A")
            else:
                m2.metric("CV Accuracy", "N/A")
            m3.metric("Vector Size", f"{X.shape[1]} floats")
            m4.metric("Samples", f"{len(images)}")
            if len(images) < 10:
                st.warning("⚠️ Training set is small (<10 samples). "
                           "Reported accuracy may not reflect real performance.")
            if is_multi:
                st.caption(f"Classes: {', '.join(head.classes_)}")

            probs = head.predict_proba(X)
            fig = go.Figure()
            for ci, cls in enumerate(head.classes_):
                fig.add_trace(go.Histogram(x=probs[:, ci], name=cls,
                                           opacity=0.7, nbinsx=20))
            fig.update_layout(title="Confidence Distribution", barmode="overlay",
                              template="plotly_dark", height=280,
                              xaxis_title="Confidence", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

            # ---- Feature Importance (RCE) ----
            st.subheader("🔍 Feature Importance")
            coefs = head.model.coef_
            feat_names = []
            for key, meta_r in REGISTRY.items():
                if active_modules.get(key, False):
                    for b in range(10):
                        feat_names.append(f"{meta_r['label']}[{b}]")

            if coefs.shape[0] == 1:
                importance = np.abs(coefs[0])
                fig_imp = go.Figure(go.Bar(
                    x=feat_names, y=importance,
                    marker_color=["#00d4ff" if "Intensity" in fn
                                  else "#ff6600" if "Sobel" in fn
                                  else "#aa00ff" for fn in feat_names]))
                fig_imp.update_layout(title="LogReg Coefficient Magnitude",
                                      template="plotly_dark", height=300,
                                      xaxis_title="Feature", yaxis_title="|Coefficient|")
            else:
                fig_imp = go.Figure()
                for ci, cls in enumerate(head.classes_):
                    if cls == "background":
                        continue
                    fig_imp.add_trace(go.Bar(
                        x=feat_names, y=np.abs(coefs[ci]),
                        name=cls, opacity=0.8))
                fig_imp.update_layout(title="LogReg Coefficients per Class",
                                      template="plotly_dark", height=300,
                                      barmode="group",
                                      xaxis_title="Feature", yaxis_title="|Coefficient|")
            st.plotly_chart(fig_imp, use_container_width=True)

            # Module-level aggregation
            module_importance = {}
            idx = 0
            for key, meta_r in REGISTRY.items():
                if active_modules.get(key, False):
                    module_importance[meta_r["label"]] = float(
                        np.abs(coefs[:, idx:idx+10]).mean())
                    idx += 10

            if module_importance:
                fig_mod = go.Figure(go.Pie(
                    labels=list(module_importance.keys()),
                    values=list(module_importance.values()),
                    hole=0.4))
                fig_mod.update_layout(title="Module Contribution (avg |coef|)",
                                      template="plotly_dark", height=280)
                st.plotly_chart(fig_mod, use_container_width=True)

            st.session_state["rce_head"] = head
            st.session_state["rce_train_acc"] = train_acc

        if "rce_head" in st.session_state:
            st.divider()
            st.subheader("Quick Predict (Crop)")
            head = st.session_state["rce_head"]
            t0 = time.perf_counter()
            vec = build_rce_vector(crop_aug, active_modules)
            label, conf = head.predict(vec)
            dt = (time.perf_counter() - t0) * 1000
            st.write(f"**{label}** — {conf:.1%} confidence — {dt:.1f} ms")


# ---------------------------------------------------------------------------
# MIDDLE — CNN Fine-Tuning
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
        images, labels, neg_short = build_training_set()
        if neg_short:
            st.warning(f"⚠️ Negative sample shortfall — training may be imbalanced.")
        backbone = meta["loader"]()

        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score

        progress = st.progress(0, text=f"Extracting {selected} features...")
        n = len(images)
        X = []
        for i, img in enumerate(images):
            X.append(backbone.get_features(img))
            progress.progress((i + 1) / n, text=f"Feature extraction: {i+1}/{n}")

        X = np.array(X)
        progress.progress(1.0, text="Fitting Logistic Regression...")

        t0 = time.perf_counter()
        try:
            head = RecognitionHead(C=cnn_C, max_iter=cnn_max_iter).fit(X, labels)
        except ValueError as e:
            st.error(f"Training failed: {e}")
            st.stop()
        train_time = time.perf_counter() - t0
        progress.progress(1.0, text="✅ Training complete!")

        preds = head.model.predict(X)
        train_acc = accuracy_score(labels, preds)

        st.success(f"Trained in **{train_time:.2f}s**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Train Accuracy", f"{train_acc:.1%}")
        if len(images) >= 6:
            n_splits = min(5, len(set(labels)))
            if n_splits >= 2:
                cv_scores = cross_val_score(head.model, X, labels,
                                           cv=min(3, len(images) // 2))
                m2.metric("CV Accuracy", f"{cv_scores.mean():.1%}",
                          delta=f"±{cv_scores.std():.1%}")
            else:
                m2.metric("CV Accuracy", "N/A")
        else:
            m2.metric("CV Accuracy", "N/A")
        m3.metric("Vector Size", f"{X.shape[1]}D")
        m4.metric("Samples", f"{len(images)}")
        if is_multi:
            st.caption(f"Classes: {', '.join(head.classes_)}")

        probs = head.predict_proba(X)
        fig = go.Figure()
        for ci, cls in enumerate(head.classes_):
            fig.add_trace(go.Histogram(x=probs[:, ci], name=cls,
                                       opacity=0.7, nbinsx=20))
        fig.update_layout(title="Confidence Distribution", barmode="overlay",
                          template="plotly_dark", height=280,
                          xaxis_title="Confidence", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        # ---- Activation Overlay (Grad-CAM style) ----
        st.subheader("🔍 Activation Overlay")
        st.caption("Highest-activation spatial regions from the hooked layer, "
                   "overlaid on the crop as a Grad-CAM–style heatmap.")
        try:
            act_maps = backbone.get_activation_maps(crop_aug, n_maps=1)
            if act_maps:
                cam = act_maps[0]
                cam_resized = cv2.resize(cam, (crop_aug.shape[1], crop_aug.shape[0]))
                cam_color = cv2.applyColorMap(
                    (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay_img = cv2.addWeighted(crop_aug, 0.5, cam_color, 0.5, 0)
                gc1, gc2 = st.columns(2)
                gc1.image(cv2.cvtColor(crop_aug, cv2.COLOR_BGR2RGB),
                          caption="Input Crop", use_container_width=True)
                gc2.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB),
                          caption="Activation Overlay", use_container_width=True)
        except Exception:
            pass

        st.session_state[f"cnn_head_{selected}"] = head
        st.session_state[f"cnn_acc_{selected}"] = train_acc

    if f"cnn_head_{selected}" in st.session_state:
        st.divider()
        st.subheader("Quick Predict (Crop)")
        backbone = meta["loader"]()
        head = st.session_state[f"cnn_head_{selected}"]
        t0 = time.perf_counter()
        feats = backbone.get_features(crop_aug)
        label, conf = head.predict(feats)
        dt = (time.perf_counter() - t0) * 1000
        st.write(f"**{label}** — {conf:.1%} confidence — {dt:.1f} ms")


# ---------------------------------------------------------------------------
# RIGHT — ORB Training
# ---------------------------------------------------------------------------
with col_orb:
    st.header("🏛️ ORB Matching")
    st.caption("Keypoint-based matching — a fundamentally different paradigm. "
               "Extracts ORB descriptors from each ROI crop and matches them "
               "against image patches using brute-force Hamming distance.")

    from src.detectors.orb import ORBDetector

    orb_dist_thresh = st.slider("Match Distance Threshold", 10, 100, 70,
                                 key="orb_dist")
    orb_min_matches = st.slider("Min Good Matches", 1, 20, 5, key="orb_min")

    if st.button("🚀 Train ORB Reference"):
        orb = ORBDetector()
        progress = st.progress(0, text="Extracting ORB descriptors...")

        orb_refs = {}
        for i, roi in enumerate(rois):
            gray = cv2.cvtColor(roi["crop_aug"], cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            kp, des = orb.orb.detectAndCompute(gray, None)
            n_feat = 0 if des is None else len(des)
            orb_refs[roi["label"]] = {
                "descriptors": des,
                "n_features":  n_feat,
                "keypoints":   kp,
                "crop":        roi["crop_aug"],
            }
            progress.progress((i + 1) / len(rois),
                              text=f"ROI {i+1}/{len(rois)}: {n_feat} features")

        progress.progress(1.0, text="✅ ORB references extracted!")

        for lbl, ref in orb_refs.items():
            if ref["keypoints"]:
                vis = cv2.drawKeypoints(ref["crop"], ref["keypoints"],
                                         None, color=(0, 255, 0))
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                         caption=f"{lbl}: {ref['n_features']} keypoints",
                         use_container_width=True)
            else:
                st.warning(f"{lbl}: No keypoints detected")

        st.session_state["orb_detector"] = orb
        st.session_state["orb_refs"] = orb_refs
        st.session_state["orb_dist_thresh"] = orb_dist_thresh
        st.session_state["orb_min_matches"] = orb_min_matches
        st.success("ORB references stored in session!")

    if "orb_refs" in st.session_state:
        st.divider()
        st.subheader("Quick Predict (Crop)")
        orb = st.session_state["orb_detector"]
        refs = st.session_state["orb_refs"]
        dt_thresh = st.session_state["orb_dist_thresh"]
        min_m = st.session_state["orb_min_matches"]

        gray = cv2.cvtColor(crop_aug, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        kp, des = orb.orb.detectAndCompute(gray, None)

        if des is not None:
            for lbl, ref in refs.items():
                if ref["descriptors"] is None:
                    st.write(f"**{lbl}:** no reference features")
                    continue
                matches = orb.bf.match(ref["descriptors"], des)
                good = [m for m in matches if m.distance < dt_thresh]
                conf = min(len(good) / max(min_m, 1), 1.0)
                verdict = lbl if len(good) >= min_m else "background"
                st.write(f"**{verdict}** — {len(good)} matches — "
                         f"{conf:.0%} confidence")
        else:
            st.write("No keypoints in test image.")


# ===========================================================================
# Bottom — Side-by-side comparison table
# ===========================================================================
st.divider()
st.subheader("📊 Training Comparison")

rows = []
rce_acc = st.session_state.get("rce_train_acc")
if rce_acc is not None:
    rows.append({"Model": "RCE", "Type": "Feature Engineering",
                 "Train Accuracy": f"{rce_acc:.1%}",
                 "Vector Size": str(sum(10 for k in active_modules if active_modules[k]))})
for name in BACKBONES:
    acc = st.session_state.get(f"cnn_acc_{name}")
    if acc is not None:
        rows.append({"Model": name, "Type": "CNN Backbone",
                     "Train Accuracy": f"{acc:.1%}",
                     "Vector Size": f"{BACKBONES[name]['dim']}D"})
if "orb_refs" in st.session_state:
    total_kp = sum(r["n_features"] for r in st.session_state["orb_refs"].values())
    rows.append({"Model": "ORB", "Type": "Keypoint Matching",
                 "Train Accuracy": "N/A (matching)",
                 "Vector Size": f"{total_kp} descriptors"})

if rows:
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("Train at least one model to see the comparison.")