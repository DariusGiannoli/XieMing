import streamlit as st
import cv2
import numpy as np
import time
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors.rce.features import REGISTRY
from src.models import BACKBONES, RecognitionHead

st.set_page_config(page_title="Real-Time Detection", layout="wide")
st.title("🎯 Real-Time Detection")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
if "pipeline_data" not in st.session_state or "crop" not in st.session_state.get("pipeline_data", {}):
    st.error("Complete **Data Lab** first (upload assets & define a crop).")
    st.stop()

assets       = st.session_state["pipeline_data"]
right_img    = assets["right"]
crop         = assets["crop"]
crop_aug     = assets.get("crop_aug", crop)
bbox         = assets.get("crop_bbox", (0, 0, crop.shape[1], crop.shape[0]))
active_mods  = st.session_state.get("active_modules", {k: True for k in REGISTRY})

x0, y0, x1, y1 = bbox
win_h, win_w = y1 - y0, x1 - x0   # window = same size as crop

rce_head = st.session_state.get("rce_head")
has_any_cnn = any(f"cnn_head_{n}" in st.session_state for n in BACKBONES)

if rce_head is None and not has_any_cnn:
    st.warning("No trained heads found. Go to **Model Tuning** and train at least one head.")
    st.stop()


# ===================================================================
#  Sliding Window Engine  (shared by both sides)
# ===================================================================
def sliding_window_detect(
    image: np.ndarray,
    feature_fn,          # callable(patch_bgr) -> 1-D np.ndarray
    head: RecognitionHead,
    stride: int,
    conf_thresh: float,
    nms_iou: float,
    progress_placeholder=None,
    live_image_placeholder=None,
):
    """
    Slide a window of size (win_h, win_w) across *image* with *stride*.
    At each position call *feature_fn* → *head.predict*.
    Returns (detections, heatmap, total_time_ms, n_windows).

    Each detection is (x, y, x+win_w, y+win_h, label, confidence).
    heatmap is a float32 array same size as image (object confidence).
    """
    H, W = image.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)
    detections = []
    t0 = time.perf_counter()

    positions = []
    for y in range(0, H - win_h + 1, stride):
        for x in range(0, W - win_w + 1, stride):
            positions.append((x, y))

    n_total = len(positions)
    if n_total == 0:
        return [], heatmap, 0.0, 0

    for idx, (x, y) in enumerate(positions):
        patch = image[y:y+win_h, x:x+win_w]
        feats = feature_fn(patch)
        label, conf = head.predict(feats)

        # Fill heatmap with object confidence
        if label == "object":
            heatmap[y:y+win_h, x:x+win_w] = np.maximum(
                heatmap[y:y+win_h, x:x+win_w], conf)
            if conf >= conf_thresh:
                detections.append((x, y, x+win_w, y+win_h, label, conf))

        # Live updates (every 5th window or last)
        if live_image_placeholder is not None and (idx % 5 == 0 or idx == n_total - 1):
            vis = image.copy()
            # Draw current scan position
            cv2.rectangle(vis, (x, y), (x+win_w, y+win_h), (255, 255, 0), 1)
            # Draw current detections
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
                (idx + 1) / n_total,
                text=f"Window {idx+1}/{n_total}")

    total_ms = (time.perf_counter() - t0) * 1000

    # --- Non-Maximum Suppression ---
    if detections:
        detections = _nms(detections, nms_iou)

    return detections, heatmap, total_ms, n_total


def _nms(dets, iou_thresh):
    """Greedy NMS on list of (x1,y1,x2,y2,label,conf)."""
    dets = sorted(dets, key=lambda d: d[5], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best, d) < iou_thresh]
    return keep


def _iou(a, b):
    """IoU between two (x1,y1,x2,y2,…) tuples."""
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    aa = (a[2]-a[0])*(a[3]-a[1])
    ab = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + ab - inter + 1e-6)


# ===================================================================
#  RCE feature function
# ===================================================================
def rce_feature_fn(patch_bgr):
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    vec = []
    for key, meta in REGISTRY.items():
        if active_mods.get(key, False):
            v, _ = meta["fn"](gray)
            vec.extend(v)
    return np.array(vec, dtype=np.float32)


# ===================================================================
#  Controls
# ===================================================================
st.subheader("Sliding Window Parameters")
p1, p2, p3 = st.columns(3)
stride = p1.slider("Stride (px)", 4, max(win_w, win_h),
                    max(win_w // 4, 4), step=2,
                    help="Lower = more windows = slower but finer")
conf_thresh = p2.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
nms_iou = p3.slider("NMS IoU Threshold", 0.1, 0.9, 0.3, 0.05)

st.caption(f"Window size: **{win_w}×{win_h} px**  |  "
           f"Right image: **{right_img.shape[1]}×{right_img.shape[0]} px**  |  "
           f"≈ {((right_img.shape[0]-win_h)//stride + 1) * ((right_img.shape[1]-win_w)//stride + 1)} windows")

st.divider()

# ===================================================================
#  Side-by-side layout
# ===================================================================
col_rce, col_cnn = st.columns(2)

# -------------------------------------------------------------------
#  LEFT — RCE Detection
# -------------------------------------------------------------------
with col_rce:
    st.header("🧬 RCE Detection")
    if rce_head is None:
        st.info("No RCE head trained. Train one in **Model Tuning**.")
    else:
        st.caption(f"Modules: {', '.join(REGISTRY[k]['label'] for k in active_mods if active_mods[k])}")
        rce_run = st.button("▶ Run RCE Scan", key="rce_run")

        rce_progress = st.empty()
        rce_live     = st.empty()
        rce_results  = st.container()

        if rce_run:
            dets, hmap, ms, nw = sliding_window_detect(
                right_img, rce_feature_fn, rce_head,
                stride, conf_thresh, nms_iou,
                progress_placeholder=rce_progress,
                live_image_placeholder=rce_live,
            )

            # Final image with boxes
            final = right_img.copy()
            for x1d, y1d, x2d, y2d, lbl, cf in dets:
                cv2.rectangle(final, (x1d, y1d), (x2d, y2d), (0, 255, 0), 2)
                cv2.putText(final, f"{cf:.0%}", (x1d, y1d - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            rce_live.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB),
                           caption="RCE — Final Detections",
                           use_container_width=True)
            rce_progress.empty()

            with rce_results:
                # Metrics
                rm1, rm2, rm3, rm4 = st.columns(4)
                rm1.metric("Detections", len(dets))
                rm2.metric("Windows", nw)
                rm3.metric("Total Time", f"{ms:.0f} ms")
                rm4.metric("Per Window", f"{ms/max(nw,1):.2f} ms")

                # Confidence heatmap
                if hmap.max() > 0:
                    hmap_color = cv2.applyColorMap(
                        (hmap / hmap.max() * 255).astype(np.uint8),
                        cv2.COLORMAP_JET)
                    blend = cv2.addWeighted(right_img, 0.5, hmap_color, 0.5, 0)
                    st.image(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB),
                             caption="RCE — Confidence Heatmap",
                             use_container_width=True)

                # Detection table
                if dets:
                    import pandas as pd
                    df = pd.DataFrame(dets, columns=["x1","y1","x2","y2","label","conf"])
                    st.dataframe(df, use_container_width=True, hide_index=True)

            st.session_state["rce_dets"] = dets
            st.session_state["rce_det_ms"] = ms

# -------------------------------------------------------------------
#  RIGHT — CNN Detection
# -------------------------------------------------------------------
with col_cnn:
    st.header("🧠 CNN Detection")

    # Find which CNN heads are trained
    trained_cnns = [n for n in BACKBONES if f"cnn_head_{n}" in st.session_state]
    if not trained_cnns:
        st.info("No CNN head trained. Train one in **Model Tuning**.")
    else:
        selected = st.selectbox("Select Model", trained_cnns, key="det_cnn_sel")
        bmeta    = BACKBONES[selected]
        backbone = bmeta["loader"]()
        head     = st.session_state[f"cnn_head_{selected}"]

        st.caption(f"Backbone: **{selected}** ({bmeta['dim']}D) — Head in session state")
        cnn_run = st.button(f"▶ Run {selected} Scan", key="cnn_run")

        cnn_progress = st.empty()
        cnn_live     = st.empty()
        cnn_results  = st.container()

        if cnn_run:
            dets, hmap, ms, nw = sliding_window_detect(
                right_img, backbone.get_features, head,
                stride, conf_thresh, nms_iou,
                progress_placeholder=cnn_progress,
                live_image_placeholder=cnn_live,
            )

            # Final image
            final = right_img.copy()
            for x1d, y1d, x2d, y2d, lbl, cf in dets:
                cv2.rectangle(final, (x1d, y1d), (x2d, y2d), (0, 0, 255), 2)
                cv2.putText(final, f"{cf:.0%}", (x1d, y1d - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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
                    blend = cv2.addWeighted(right_img, 0.5, hmap_color, 0.5, 0)
                    st.image(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB),
                             caption=f"{selected} — Confidence Heatmap",
                             use_container_width=True)

                if dets:
                    import pandas as pd
                    df = pd.DataFrame(dets, columns=["x1","y1","x2","y2","label","conf"])
                    st.dataframe(df, use_container_width=True, hide_index=True)

            st.session_state["cnn_dets"] = dets
            st.session_state["cnn_det_ms"] = ms


# ===================================================================
#  Bottom — Comparison (if both have run)
# ===================================================================
rce_dets = st.session_state.get("rce_dets")
cnn_dets = st.session_state.get("cnn_dets")

if rce_dets is not None and cnn_dets is not None:
    st.divider()
    st.subheader("📊 Side-by-Side Comparison")

    import pandas as pd
    comp = pd.DataFrame({
        "Metric": ["Detections", "Best Confidence", "Total Time (ms)"],
        "RCE": [
            len(rce_dets),
            f"{max((d[5] for d in rce_dets), default=0):.1%}",
            f"{st.session_state.get('rce_det_ms', 0):.0f}",
        ],
        "CNN": [
            len(cnn_dets),
            f"{max((d[5] for d in cnn_dets), default=0):.1%}",
            f"{st.session_state.get('cnn_det_ms', 0):.0f}",
        ],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    # Overlay both on one image
    overlay = right_img.copy()
    for x1d, y1d, x2d, y2d, _, cf in rce_dets:
        cv2.rectangle(overlay, (x1d, y1d), (x2d, y2d), (0, 255, 0), 2)
        cv2.putText(overlay, f"RCE {cf:.0%}", (x1d, y1d - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    for x1d, y1d, x2d, y2d, _, cf in cnn_dets:
        cv2.rectangle(overlay, (x1d, y1d), (x2d, y2d), (0, 0, 255), 2)
        cv2.putText(overlay, f"CNN {cf:.0%}", (x1d, y2d + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
             caption="Green = RCE  |  Blue = CNN",
             use_container_width=True)
