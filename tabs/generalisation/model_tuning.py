"""Generalisation Model Tuning — Stage 3 of the Generalisation pipeline."""

import streamlit as st
import cv2
import numpy as np
import time
import plotly.graph_objects as go

from src.detectors.rce.features import REGISTRY
from src.models import BACKBONES, RecognitionHead
from src.utils import build_rce_vector


def render():
    st.title("⚙️ Model Tuning: Train & Compare")

    pipe = st.session_state.get("gen_pipeline")
    if not pipe or "crop" not in pipe:
        st.error("Please complete the **Data Lab** first.")
        st.stop()

    crop      = pipe["crop"]
    crop_aug  = pipe.get("crop_aug", crop)
    train_img = pipe["train_image"]
    bbox      = pipe.get("crop_bbox", (0, 0, crop.shape[1], crop.shape[0]))
    rois      = pipe.get("rois", [{"label": "object", "bbox": bbox,
                                    "crop": crop, "crop_aug": crop_aug}])
    active_modules = pipe.get("active_modules", {k: True for k in REGISTRY})
    is_multi = len(rois) > 1

    def build_training_set():
        images, labels = [], []
        for roi in rois:
            images.append(roi["crop"]); labels.append(roi["label"])
            images.append(roi["crop_aug"]); labels.append(roi["label"])
        all_bboxes = [roi["bbox"] for roi in rois]
        H, W = train_img.shape[:2]
        x0r, y0r, x1r, y1r = rois[0]["bbox"]
        ch, cw = y1r - y0r, x1r - x0r
        rng = np.random.default_rng(42)
        n_neg_target = len(images) * 2
        attempts, negatives = 0, []
        while len(negatives) < n_neg_target and attempts < 300:
            rx = rng.integers(0, max(W - cw, 1))
            ry = rng.integers(0, max(H - ch, 1))
            overlaps = any(rx < bx1 and rx + cw > bx0 and ry < by1 and ry + ch > by0
                           for bx0, by0, bx1, by1 in all_bboxes)
            if overlaps:
                attempts += 1; continue
            patch = train_img[ry:ry+ch, rx:rx+cw]
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                negatives.append(patch)
            attempts += 1
        images.extend(negatives)
        labels.extend(["background"] * len(negatives))
        return images, labels, len(negatives) < n_neg_target // 2

    # Show training data
    st.subheader("Training Data (from Data Lab)")
    if is_multi:
        st.caption(f"**{len(rois)} classes** defined.")
        roi_cols = st.columns(min(len(rois), 4))
        for i, roi in enumerate(rois):
            with roi_cols[i % len(roi_cols)]:
                st.image(cv2.cvtColor(roi["crop"], cv2.COLOR_BGR2RGB),
                         caption=f"✅ {roi['label']}", width=140)
    else:
        td1, td2 = st.columns(2)
        td1.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                  caption="Original Crop (positive)", width=180)
        td2.image(cv2.cvtColor(crop_aug, cv2.COLOR_BGR2RGB),
                  caption="Augmented Crop (positive)", width=180)
    st.divider()

    col_rce, col_cnn, col_orb = st.columns(3)

    # -------------------------------------------------------------------
    # RCE Training
    # -------------------------------------------------------------------
    with col_rce:
        st.header("🧬 RCE Training")
        active_names = [REGISTRY[k]["label"] for k in active_modules if active_modules[k]]
        if not active_names:
            st.error("No RCE modules selected.")
        else:
            st.write(f"**Active modules:** {', '.join(active_names)}")
            rce_C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, step=0.01, key="gen_rce_c")
            rce_max_iter = st.slider("Max Iterations", 100, 5000, 1000, step=100, key="gen_rce_iter")

            if st.button("🚀 Train RCE Head", key="gen_train_rce"):
                images, labels, neg_short = build_training_set()
                if neg_short:
                    st.warning("⚠️ Few negatives collected.")
                from sklearn.metrics import accuracy_score
                from sklearn.model_selection import cross_val_score
                progress = st.progress(0, text="Extracting RCE features...")
                n = len(images)
                X = [build_rce_vector(img, active_modules) for i, img in enumerate(images)
                     if not progress.progress((i + 1) / n, text=f"Feature extraction: {i+1}/{n}") or True]
                # rebuild X properly
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
                    st.error(f"Training failed: {e}"); st.stop()
                train_time = time.perf_counter() - t0
                progress.progress(1.0, text="✅ Training complete!")

                train_acc = accuracy_score(labels, head.model.predict(X))
                st.success(f"Trained in **{train_time:.2f}s**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Train Accuracy", f"{train_acc:.1%}")
                if len(images) >= 6:
                    n_splits = min(5, len(set(labels)))
                    if n_splits >= 2:
                        cv_scores = cross_val_score(head.model, X, labels, cv=min(3, len(images) // 2))
                        m2.metric("CV Accuracy", f"{cv_scores.mean():.1%}", delta=f"±{cv_scores.std():.1%}")
                    else:
                        m2.metric("CV Accuracy", "N/A")
                else:
                    m2.metric("CV Accuracy", "N/A")
                m3.metric("Vector Size", f"{X.shape[1]} floats")
                m4.metric("Samples", f"{len(images)}")

                # Feature importance
                coefs = head.model.coef_
                feat_names = []
                for key, meta_r in REGISTRY.items():
                    if active_modules.get(key, False):
                        for b in range(10):
                            feat_names.append(f"{meta_r['label']}[{b}]")
                if coefs.shape[0] == 1:
                    fig_imp = go.Figure(go.Bar(x=feat_names, y=np.abs(coefs[0])))
                    fig_imp.update_layout(title="LogReg Coefficient Magnitude",
                                          template="plotly_dark", height=300)
                else:
                    fig_imp = go.Figure()
                    for ci, cls in enumerate(head.classes_):
                        if cls == "background": continue
                        fig_imp.add_trace(go.Bar(x=feat_names, y=np.abs(coefs[ci]),
                                                  name=cls, opacity=0.8))
                    fig_imp.update_layout(title="Coefficients per Class",
                                          template="plotly_dark", height=300, barmode="group")
                st.plotly_chart(fig_imp, use_container_width=True)

                pipe["rce_head"] = head
                pipe["rce_train_acc"] = train_acc
                st.session_state["gen_pipeline"] = pipe

            if pipe.get("rce_head"):
                st.divider()
                st.subheader("Quick Predict")
                head = pipe["rce_head"]
                vec = build_rce_vector(crop_aug, active_modules)
                label, conf = head.predict(vec)
                st.write(f"**{label}** — {conf:.1%} confidence")

    # -------------------------------------------------------------------
    # CNN Fine-Tuning
    # -------------------------------------------------------------------
    with col_cnn:
        st.header("🧠 CNN Fine-Tuning")
        selected = st.selectbox("Select Model", list(BACKBONES.keys()), key="gen_mt_cnn")
        meta = BACKBONES[selected]
        st.caption(f"Backbone: **{meta['dim']}D** → Logistic Regression")
        cnn_C = st.slider("Regularization (C) ", 0.01, 10.0, 1.0, step=0.01, key="gen_cnn_c")
        cnn_max_iter = st.slider("Max Iterations ", 100, 5000, 1000, step=100, key="gen_cnn_iter")

        if st.button(f"🚀 Train {selected} Head", key="gen_train_cnn"):
            images, labels, neg_short = build_training_set()
            backbone = meta["loader"]()
            from sklearn.metrics import accuracy_score
            progress = st.progress(0, text=f"Extracting {selected} features...")
            n = len(images)
            X = []
            for i, img in enumerate(images):
                X.append(backbone.get_features(img))
                progress.progress((i + 1) / n, text=f"Feature extraction: {i+1}/{n}")
            X = np.array(X)
            progress.progress(1.0, text="Fitting...")

            t0 = time.perf_counter()
            try:
                head = RecognitionHead(C=cnn_C, max_iter=cnn_max_iter).fit(X, labels)
            except ValueError as e:
                st.error(f"Training failed: {e}"); st.stop()
            train_time = time.perf_counter() - t0
            progress.progress(1.0, text="✅ Done!")

            train_acc = accuracy_score(labels, head.model.predict(X))
            st.success(f"Trained in **{train_time:.2f}s**")
            m1, m2 = st.columns(2)
            m1.metric("Train Accuracy", f"{train_acc:.1%}")
            m2.metric("Samples", f"{len(images)}")

            pipe[f"cnn_head_{selected}"] = head
            pipe[f"cnn_acc_{selected}"] = train_acc
            st.session_state["gen_pipeline"] = pipe

        if pipe.get(f"cnn_head_{selected}"):
            st.divider()
            st.subheader("Quick Predict")
            backbone = meta["loader"]()
            head = pipe[f"cnn_head_{selected}"]
            feats = backbone.get_features(crop_aug)
            label, conf = head.predict(feats)
            st.write(f"**{label}** — {conf:.1%} confidence")

    # -------------------------------------------------------------------
    # ORB Training
    # -------------------------------------------------------------------
    with col_orb:
        st.header("🏛️ ORB Matching")
        from src.detectors.orb import ORBDetector
        orb_dist_thresh = st.slider("Match Distance Threshold", 10, 100, 70, key="gen_orb_dist")
        orb_min_matches = st.slider("Min Good Matches", 1, 20, 5, key="gen_orb_min")

        if st.button("🚀 Train ORB Reference", key="gen_train_orb"):
            orb = ORBDetector()
            orb_refs = {}
            for i, roi in enumerate(rois):
                gray = cv2.cvtColor(roi["crop_aug"], cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                kp, des = orb.orb.detectAndCompute(gray, None)
                n_feat = 0 if des is None else len(des)
                orb_refs[roi["label"]] = {"descriptors": des, "n_features": n_feat,
                                           "keypoints": kp, "crop": roi["crop_aug"]}

            for lbl, ref in orb_refs.items():
                if ref["keypoints"]:
                    vis = cv2.drawKeypoints(ref["crop"], ref["keypoints"], None, color=(0, 255, 0))
                    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                             caption=f"{lbl}: {ref['n_features']} keypoints",
                             use_container_width=True)

            pipe["orb_detector"] = orb
            pipe["orb_refs"] = orb_refs
            pipe["orb_dist_thresh"] = orb_dist_thresh
            pipe["orb_min_matches"] = orb_min_matches
            st.session_state["gen_pipeline"] = pipe
            st.success("ORB references stored!")

    # Comparison Table
    st.divider()
    st.subheader("📊 Training Comparison")
    rows = []
    if pipe.get("rce_train_acc") is not None:
        rows.append({"Model": "RCE", "Type": "Feature Engineering",
                     "Train Accuracy": f"{pipe['rce_train_acc']:.1%}"})
    for name in BACKBONES:
        acc = pipe.get(f"cnn_acc_{name}")
        if acc is not None:
            rows.append({"Model": name, "Type": "CNN Backbone",
                         "Train Accuracy": f"{acc:.1%}"})
    if pipe.get("orb_refs"):
        rows.append({"Model": "ORB", "Type": "Keypoint Matching",
                     "Train Accuracy": "N/A"})
    if rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Train at least one model to see the comparison.")
