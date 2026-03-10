"""Stereo Detection — Stage 5 of the Stereo + Depth pipeline.

CRITICAL: Detection runs on the TEST (right) image.
          Training was done on the TRAIN (left) image.
          This enforces the data-leakage fix.
"""

import streamlit as st
import cv2
import numpy as np
import time
import plotly.graph_objects as go

from src.detectors.rce.features import REGISTRY
from src.models import BACKBONES, RecognitionHead
from src.utils import build_rce_vector
from src.localization import nms as _nms


CLASS_COLORS = [(0,255,0),(0,0,255),(255,165,0),(255,0,255),(0,255,255),
                (128,255,0),(255,128,0),(0,128,255)]


def sliding_window_detect(image, feature_fn, head, win_h, win_w,
                           stride, conf_thresh, nms_iou,
                           progress_placeholder=None,
                           live_image_placeholder=None):
    H, W = image.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)
    detections = []
    t0 = time.perf_counter()

    positions = [(x, y)
                 for y in range(0, H - win_h + 1, stride)
                 for x in range(0, W - win_w + 1, stride)]
    n_total = len(positions)
    if n_total == 0:
        return [], heatmap, 0.0, 0

    for idx, (x, y) in enumerate(positions):
        patch = image[y:y+win_h, x:x+win_w]
        feats = feature_fn(patch)
        label, conf = head.predict(feats)

        if label != "background":
            heatmap[y:y+win_h, x:x+win_w] = np.maximum(
                heatmap[y:y+win_h, x:x+win_w], conf)
            if conf >= conf_thresh:
                detections.append((x, y, x+win_w, y+win_h, label, conf))

        if live_image_placeholder is not None and (idx % 5 == 0 or idx == n_total - 1):
            vis = image.copy()
            cv2.rectangle(vis, (x, y), (x+win_w, y+win_h), (255, 255, 0), 1)
            for dx, dy, dx2, dy2, dl, dc in detections:
                cv2.rectangle(vis, (dx, dy), (dx2, dy2), (0, 255, 0), 2)
                cv2.putText(vis, f"{dc:.0%}", (dx, dy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            live_image_placeholder.image(
                cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                caption=f"Scanning… {idx+1}/{n_total}",
                use_container_width=True)

        if progress_placeholder is not None:
            progress_placeholder.progress(
                (idx + 1) / n_total, text=f"Window {idx+1}/{n_total}")

    total_ms = (time.perf_counter() - t0) * 1000
    if detections:
        detections = _nms(detections, nms_iou)
    return detections, heatmap, total_ms, n_total


def render():
    st.title("🎯 Real-Time Detection")

    pipe = st.session_state.get("stereo_pipeline")
    if not pipe or "crop" not in pipe:
        st.error("Complete **Data Lab** first (upload assets & define a crop).")
        st.stop()

    # CRITICAL: detect on TEST image (right), not TRAIN image (left)
    test_img     = pipe["test_image"]
    crop         = pipe["crop"]
    crop_aug     = pipe.get("crop_aug", crop)
    bbox         = pipe.get("crop_bbox", (0, 0, crop.shape[1], crop.shape[0]))
    rois         = pipe.get("rois", [{"label": "object", "bbox": bbox,
                                       "crop": crop, "crop_aug": crop_aug}])
    active_mods  = pipe.get("active_modules", {k: True for k in REGISTRY})

    x0, y0, x1, y1 = bbox
    win_h, win_w = y1 - y0, x1 - x0

    if win_h <= 0 or win_w <= 0:
        st.error("Invalid window size from crop bbox.")
        st.stop()

    rce_head = pipe.get("rce_head")
    has_any_cnn = any(f"cnn_head_{n}" in pipe for n in BACKBONES)
    has_orb = pipe.get("orb_refs") is not None

    if rce_head is None and not has_any_cnn and not has_orb:
        st.warning("No trained heads found. Go to **Model Tuning** first.")
        st.stop()

    def rce_feature_fn(patch_bgr):
        return build_rce_vector(patch_bgr, active_mods)

    # Controls
    st.subheader("Sliding Window Parameters")
    p1, p2, p3 = st.columns(3)
    stride = p1.slider("Stride (px)", 4, max(win_w // 2, 4),
                        max(win_w // 4, 4), step=2, key="stereo_det_stride")
    conf_thresh = p2.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.05,
                             key="stereo_det_conf")
    nms_iou = p3.slider("NMS IoU Threshold", 0.1, 0.9, 0.3, 0.05,
                         key="stereo_det_nms")

    st.caption(f"Window size: **{win_w}×{win_h} px**  |  "
               f"Test image: **{test_img.shape[1]}×{test_img.shape[0]} px**  |  "
               f"≈ {((test_img.shape[0]-win_h)//stride + 1) * ((test_img.shape[1]-win_w)//stride + 1)} windows")
    st.divider()

    col_rce, col_cnn, col_orb = st.columns(3)

    # -------------------------------------------------------------------
    # RCE Detection
    # -------------------------------------------------------------------
    with col_rce:
        st.header("🧬 RCE Detection")
        if rce_head is None:
            st.info("No RCE head trained.")
        else:
            st.caption(f"Modules: {', '.join(REGISTRY[k]['label'] for k in active_mods if active_mods[k])}")
            rce_run = st.button("▶ Run RCE Scan", key="stereo_rce_run")
            rce_progress = st.empty()
            rce_live = st.empty()
            rce_results = st.container()

            if rce_run:
                dets, hmap, ms, nw = sliding_window_detect(
                    test_img, rce_feature_fn, rce_head, win_h, win_w,
                    stride, conf_thresh, nms_iou,
                    progress_placeholder=rce_progress,
                    live_image_placeholder=rce_live)

                final = test_img.copy()
                class_labels = sorted(set(d[4] for d in dets)) if dets else []
                for x1d, y1d, x2d, y2d, lbl, cf in dets:
                    ci = class_labels.index(lbl) if lbl in class_labels else 0
                    clr = CLASS_COLORS[ci % len(CLASS_COLORS)]
                    cv2.rectangle(final, (x1d, y1d), (x2d, y2d), clr, 2)
                    cv2.putText(final, f"{lbl} {cf:.0%}", (x1d, y1d - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1)
                rce_live.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB),
                               caption="RCE — Final Detections",
                               use_container_width=True)
                rce_progress.empty()

                with rce_results:
                    rm1, rm2, rm3, rm4 = st.columns(4)
                    rm1.metric("Detections", len(dets))
                    rm2.metric("Windows", nw)
                    rm3.metric("Total Time", f"{ms:.0f} ms")
                    rm4.metric("Per Window", f"{ms/max(nw,1):.2f} ms")

                    if hmap.max() > 0:
                        hmap_color = cv2.applyColorMap(
                            (hmap / hmap.max() * 255).astype(np.uint8),
                            cv2.COLORMAP_JET)
                        blend = cv2.addWeighted(test_img, 0.5, hmap_color, 0.5, 0)
                        st.image(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB),
                                 caption="RCE — Confidence Heatmap",
                                 use_container_width=True)

                    if dets:
                        import pandas as pd
                        df = pd.DataFrame(dets, columns=["x1","y1","x2","y2","label","conf"])
                        st.dataframe(df, use_container_width=True, hide_index=True)

                pipe["rce_dets"] = dets
                pipe["rce_det_ms"] = ms
                st.session_state["stereo_pipeline"] = pipe

    # -------------------------------------------------------------------
    # CNN Detection
    # -------------------------------------------------------------------
    with col_cnn:
        st.header("🧠 CNN Detection")
        trained_cnns = [n for n in BACKBONES if f"cnn_head_{n}" in pipe]
        if not trained_cnns:
            st.info("No CNN head trained.")
        else:
            selected = st.selectbox("Select Model", trained_cnns,
                                    key="stereo_det_cnn_sel")
            bmeta    = BACKBONES[selected]
            backbone = bmeta["loader"]()
            head     = pipe[f"cnn_head_{selected}"]

            st.caption(f"Backbone: **{selected}** ({bmeta['dim']}D)")
            cnn_run = st.button(f"▶ Run {selected} Scan", key="stereo_cnn_run")
            cnn_progress = st.empty()
            cnn_live = st.empty()
            cnn_results = st.container()

            if cnn_run:
                dets, hmap, ms, nw = sliding_window_detect(
                    test_img, backbone.get_features, head, win_h, win_w,
                    stride, conf_thresh, nms_iou,
                    progress_placeholder=cnn_progress,
                    live_image_placeholder=cnn_live)

                final = test_img.copy()
                class_labels = sorted(set(d[4] for d in dets)) if dets else []
                for x1d, y1d, x2d, y2d, lbl, cf in dets:
                    ci = class_labels.index(lbl) if lbl in class_labels else 0
                    clr = CLASS_COLORS[ci % len(CLASS_COLORS)]
                    cv2.rectangle(final, (x1d, y1d), (x2d, y2d), clr, 2)
                    cv2.putText(final, f"{lbl} {cf:.0%}", (x1d, y1d - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1)
                cnn_live.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB),
                               caption=f"{selected} — Final Detections",
                               use_container_width=True)
                cnn_progress.empty()

                with cnn_results:
                    cm1, cm2, cm3, cm4 = st.columns(4)
                    cm1.metric("Detections", len(dets))
                    cm2.metric("Windows", nw)
                    cm3.metric("Total Time", f"{ms:.0f} ms")
                    cm4.metric("Per Window", f"{ms/max(nw,1):.2f} ms")

                    if hmap.max() > 0:
                        hmap_color = cv2.applyColorMap(
                            (hmap / hmap.max() * 255).astype(np.uint8),
                            cv2.COLORMAP_JET)
                        blend = cv2.addWeighted(test_img, 0.5, hmap_color, 0.5, 0)
                        st.image(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB),
                                 caption=f"{selected} — Confidence Heatmap",
                                 use_container_width=True)

                    if dets:
                        import pandas as pd
                        df = pd.DataFrame(dets, columns=["x1","y1","x2","y2","label","conf"])
                        st.dataframe(df, use_container_width=True, hide_index=True)

                pipe["cnn_dets"] = dets
                pipe["cnn_det_ms"] = ms
                st.session_state["stereo_pipeline"] = pipe

    # -------------------------------------------------------------------
    # ORB Detection
    # -------------------------------------------------------------------
    with col_orb:
        st.header("🏛️ ORB Detection")
        if not has_orb:
            st.info("No ORB reference trained.")
        else:
            orb_det  = pipe["orb_detector"]
            orb_refs = pipe["orb_refs"]
            dt_thresh = pipe.get("orb_dist_thresh", 70)
            min_m     = pipe.get("orb_min_matches", 5)
            st.caption(f"References: {', '.join(orb_refs.keys())}  |  "
                       f"dist<{dt_thresh}, min {min_m} matches")
            orb_run = st.button("▶ Run ORB Scan", key="stereo_orb_run")
            orb_progress = st.empty()
            orb_live = st.empty()
            orb_results = st.container()

            if orb_run:
                H, W = test_img.shape[:2]
                positions = [(x, y)
                             for y in range(0, H - win_h + 1, stride)
                             for x in range(0, W - win_w + 1, stride)]
                n_total = len(positions)
                heatmap = np.zeros((H, W), dtype=np.float32)
                detections = []
                t0 = time.perf_counter()
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

                for idx, (px, py) in enumerate(positions):
                    patch = test_img[py:py+win_h, px:px+win_w]
                    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    gray = clahe.apply(gray)
                    kp, des = orb_det.orb.detectAndCompute(gray, None)

                    if des is not None:
                        best_label, best_conf = "background", 0.0
                        for lbl, ref in orb_refs.items():
                            if ref["descriptors"] is None:
                                continue
                            matches = orb_det.bf.match(ref["descriptors"], des)
                            good = [m for m in matches if m.distance < dt_thresh]
                            conf = min(len(good) / max(min_m, 1), 1.0)
                            if len(good) >= min_m and conf > best_conf:
                                best_label, best_conf = lbl, conf

                        if best_label != "background":
                            heatmap[py:py+win_h, px:px+win_w] = np.maximum(
                                heatmap[py:py+win_h, px:px+win_w], best_conf)
                            if best_conf >= conf_thresh:
                                detections.append(
                                    (px, py, px+win_w, py+win_h, best_label, best_conf))

                    if idx % 5 == 0 or idx == n_total - 1:
                        orb_progress.progress((idx+1)/n_total,
                                              text=f"Window {idx+1}/{n_total}")

                total_ms = (time.perf_counter() - t0) * 1000
                if detections:
                    detections = _nms(detections, nms_iou)

                final = test_img.copy()
                cls_labels = sorted(set(d[4] for d in detections)) if detections else []
                for x1d, y1d, x2d, y2d, lbl, cf in detections:
                    ci = cls_labels.index(lbl) if lbl in cls_labels else 0
                    clr = CLASS_COLORS[ci % len(CLASS_COLORS)]
                    cv2.rectangle(final, (x1d, y1d), (x2d, y2d), clr, 2)
                    cv2.putText(final, f"{lbl} {cf:.0%}", (x1d, y1d - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1)
                orb_live.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB),
                               caption="ORB — Final Detections",
                               use_container_width=True)
                orb_progress.empty()

                with orb_results:
                    om1, om2, om3, om4 = st.columns(4)
                    om1.metric("Detections", len(detections))
                    om2.metric("Windows", n_total)
                    om3.metric("Total Time", f"{total_ms:.0f} ms")
                    om4.metric("Per Window", f"{total_ms/max(n_total,1):.2f} ms")

                    if heatmap.max() > 0:
                        hmap_color = cv2.applyColorMap(
                            (heatmap / heatmap.max() * 255).astype(np.uint8),
                            cv2.COLORMAP_JET)
                        blend = cv2.addWeighted(test_img, 0.5, hmap_color, 0.5, 0)
                        st.image(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB),
                                 caption="ORB — Confidence Heatmap",
                                 use_container_width=True)

                    if detections:
                        import pandas as pd
                        df = pd.DataFrame(detections,
                                          columns=["x1","y1","x2","y2","label","conf"])
                        st.dataframe(df, use_container_width=True, hide_index=True)

                pipe["orb_dets"] = detections
                pipe["orb_det_ms"] = total_ms
                st.session_state["stereo_pipeline"] = pipe

    # ===================================================================
    # Bottom — Comparison
    # ===================================================================
    rce_dets = pipe.get("rce_dets")
    cnn_dets = pipe.get("cnn_dets")
    orb_dets = pipe.get("orb_dets")

    methods = {}
    if rce_dets is not None:
        methods["RCE"] = (rce_dets, pipe.get("rce_det_ms", 0), (0,255,0))
    if cnn_dets is not None:
        methods["CNN"] = (cnn_dets, pipe.get("cnn_det_ms", 0), (0,0,255))
    if orb_dets is not None:
        methods["ORB"] = (orb_dets, pipe.get("orb_det_ms", 0), (255,165,0))

    if len(methods) >= 2:
        st.divider()
        st.subheader("📊 Side-by-Side Comparison")
        import pandas as pd
        comp = {"Metric": ["Detections", "Best Confidence", "Total Time (ms)"]}
        for name, (dets, ms, _) in methods.items():
            comp[name] = [
                len(dets),
                f"{max((d[5] for d in dets), default=0):.1%}",
                f"{ms:.0f}",
            ]
        st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

        overlay = test_img.copy()
        for name, (dets, _, clr) in methods.items():
            for x1d, y1d, x2d, y2d, lbl, cf in dets:
                cv2.rectangle(overlay, (x1d, y1d), (x2d, y2d), clr, 2)
                cv2.putText(overlay, f"{name}:{lbl} {cf:.0%}", (x1d, y1d - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, clr, 1)
        legend = " | ".join(f"{n}={'green' if c==(0,255,0) else 'blue' if c==(0,0,255) else 'orange'}"
                            for n, (_, _, c) in methods.items())
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption=legend, use_container_width=True)
