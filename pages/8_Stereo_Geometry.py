import streamlit as st
import cv2
import numpy as np
import re
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Stereo Geometry", layout="wide")
st.title("📐 Stereo Geometry: Distance Estimation")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
if "pipeline_data" not in st.session_state or "left" not in st.session_state.get("pipeline_data", {}):
    st.error("Complete **Data Lab** first.")
    st.stop()

assets    = st.session_state["pipeline_data"]
img_l     = assets["left"]
img_r     = assets["right"]
gt_left   = assets.get("gt_left")       # float32 disparity map from PFM
gt_right  = assets.get("gt_right")
conf_raw  = assets.get("conf_raw", "")
crop_bbox = assets.get("crop_bbox")      # (x0, y0, x1, y1) on LEFT image

rce_dets = st.session_state.get("rce_dets", [])
cnn_dets = st.session_state.get("cnn_dets", [])


# ===================================================================
#  Parse Middlebury-style camera config
# ===================================================================
def parse_config(text: str) -> dict:
    """
    Parse a Middlebury .txt / .conf calibration file.
    Expected keys: cam0, cam1, doffs, baseline, width, height, ndisp, vmin, vmax
    cam0 / cam1 are 3×3 matrices in bracket notation: [f 0 cx; 0 f cy; 0 0 1]
    """
    params = {}
    if not text or not text.strip():
        return params
    for line in text.strip().splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if "[" in val:
            nums = list(map(float, re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)))
            params[key] = np.array(nums).reshape(3, 3) if len(nums) == 9 else nums
        else:
            try:
                params[key] = float(val)
            except ValueError:
                params[key] = val
    return params


calib = parse_config(conf_raw)

# Extract intrinsics
cam0 = calib.get("cam0")
focal    = float(cam0[0, 0]) if isinstance(cam0, np.ndarray) and cam0.shape == (3, 3) else 0.0
doffs    = float(calib.get("doffs", 0.0))
baseline = float(calib.get("baseline", 1.0))
ndisp    = int(calib.get("ndisp", 128))

if focal <= 0:
    st.error("❌ Focal length is **0** — the camera config is missing or malformed. "
            "Depth estimation cannot proceed. Return to **Data Lab** and upload a valid "
            "Middlebury camera config.")
    st.stop()

if focal > 10000:
    st.error(f"❌ Focal length ({focal:.0f} px) is suspiciously large. "
             "Check your camera config file.")
    st.stop()

if baseline <= 0 or baseline > 1000:
    st.error(f"❌ Invalid baseline ({baseline:.1f}). Expected 1–1000 mm.")
    st.stop()

st.subheader("Camera Calibration")
cc1, cc2, cc3, cc4 = st.columns(4)
cc1.metric("Focal Length (px)", f"{focal:.1f}")
cc2.metric("Baseline (mm)", f"{baseline:.1f}")
cc3.metric("Doffs (px)", f"{doffs:.2f}")
cc4.metric("ndisp", str(ndisp))

with st.expander("Full Calibration"):
    st.json({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in calib.items()})

st.divider()


# ===================================================================
#  Image-size validation
# ===================================================================
if img_l.shape[:2] != img_r.shape[:2]:
    st.error(f"Left ({img_l.shape[1]}×{img_l.shape[0]}) and right "
             f"({img_r.shape[1]}×{img_r.shape[0]}) images must be the same size.")
    st.stop()


# ===================================================================
#  Step 1 — Compute Disparity Map
# ===================================================================
st.subheader("Step 1: Disparity Map (StereoSGBM)")

sc1, sc2, sc3 = st.columns(3)
block_size  = sc1.slider("Block Size", 3, 21, 5, step=2)
p1_mult     = sc2.slider("P1 multiplier", 1, 32, 8)
p2_mult     = sc3.slider("P2 multiplier", 1, 128, 32)


@st.cache_data
def compute_disparity(_left, _right, ndisp, block_size, p1m, p2m):
    """StereoSGBM disparity.  _left/_right are un-hashed (numpy arrays)."""
    gray_l = cv2.cvtColor(_left,  cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(_right, cv2.COLOR_BGR2GRAY)

    nd = max(16, (ndisp // 16) * 16)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=nd,
        blockSize=block_size,
        P1=p1m * 1 * block_size ** 2,
        P2=p2m * 1 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0


with st.spinner("Computing disparity…"):
    try:
        disp = compute_disparity(img_l, img_r, ndisp, block_size, p1_mult, p2_mult)
    except cv2.error as e:
        st.error(f"StereoSGBM failed: {e}")
        st.stop()

# Visualize disparity
disp_vis = np.clip(disp, 0, None)
disp_max = disp_vis.max() if disp_vis.max() > 0 else 1.0
disp_norm = (disp_vis / disp_max * 255).astype(np.uint8)
disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_INFERNO)

dc1, dc2 = st.columns(2)
dc1.image(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB), caption="Left Image", use_container_width=True)
dc2.image(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB), caption="Disparity Map (SGBM)", use_container_width=True)


# ===================================================================
#  Step 2 — Depth Map from Disparity
# ===================================================================
st.divider()
st.subheader("Step 2: Depth Map from Disparity")

st.latex(r"Z = \frac{f \times B}{d + d_{\text{offs}}}")
st.caption("Z = depth (mm), f = focal length (px), B = baseline (mm), d = disparity (px), d_offs = optical center offset (px)")

# Compute depth from disparity
valid = (disp + doffs) > 0
depth_map = np.zeros_like(disp)
if focal > 0:
    depth_map[valid] = (focal * baseline) / (disp[valid] + doffs)

# Visualize
depth_vis = depth_map.copy()
finite = depth_vis[depth_vis > 0]
if len(finite) > 0:
    clip_max = np.percentile(finite, 98)
    depth_vis = np.clip(depth_vis, 0, clip_max)
    depth_norm = (depth_vis / clip_max * 255).astype(np.uint8)
else:
    depth_norm = np.zeros_like(depth_map, dtype=np.uint8)

depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

zc1, zc2 = st.columns(2)
zc1.image(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB),
          caption="Estimated Depth (SGBM)", use_container_width=True)

# Ground truth comparison
if gt_left is not None:
    gt_vis = gt_left.copy()
    gt_finite = gt_vis[np.isfinite(gt_vis) & (gt_vis > 0)]
    if len(gt_finite) > 0:
        gt_clip = np.percentile(gt_finite, 98)
        gt_vis = np.clip(np.nan_to_num(gt_vis, nan=0), 0, gt_clip)
        gt_norm = (gt_vis / gt_clip * 255).astype(np.uint8)
    else:
        gt_norm = np.zeros_like(gt_vis, dtype=np.uint8)
    gt_color = cv2.applyColorMap(gt_norm, cv2.COLORMAP_TURBO)
    zc2.image(cv2.cvtColor(gt_color, cv2.COLOR_BGR2RGB),
              caption="Ground Truth Disparity (from PFM)", use_container_width=True)


# ===================================================================
#  Step 3 — Error Map (SGBM vs Ground Truth)
# ===================================================================
if gt_left is not None:
    st.divider()
    st.subheader("Step 3: Error Analysis (SGBM vs Ground Truth)")

    gt_disp = gt_left   # Middlebury standard: PFM = disparity map

    # Ensure GT and SGBM disparity have the same shape
    if gt_disp.shape[:2] != disp.shape[:2]:
        st.warning(
            f"Ground truth shape ({gt_disp.shape[1]}×{gt_disp.shape[0]}) differs from "
            f"disparity shape ({disp.shape[1]}×{disp.shape[0]}). Resizing GT to match."
        )
        gt_disp = cv2.resize(gt_disp, (disp.shape[1], disp.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    gt_valid = np.isfinite(gt_disp) & (gt_disp > 0)
    both_valid = valid & gt_valid

    if both_valid.any():
        # Disparity error
        disp_err = np.abs(disp - gt_disp)
        disp_err[~both_valid] = 0

        # Stats
        err_vals = disp_err[both_valid]
        mae  = float(np.mean(err_vals))
        rmse = float(np.sqrt(np.mean(err_vals ** 2)))
        bad_2 = float(np.mean(err_vals > 2.0)) * 100

        em1, em2, em3 = st.columns(3)
        em1.metric("MAE (px)", f"{mae:.2f}")
        em2.metric("RMSE (px)", f"{rmse:.2f}")
        em3.metric("Bad-2.0 (%)", f"{bad_2:.1f}%")

        # Error heatmap
        err_clip = np.clip(disp_err, 0, 10)
        err_norm = (err_clip / 10 * 255).astype(np.uint8)
        err_color = cv2.applyColorMap(err_norm, cv2.COLORMAP_HOT)
        st.image(cv2.cvtColor(err_color, cv2.COLOR_BGR2RGB),
                 caption="Disparity Error Map (red = high error, clipped at 10 px)",
                 use_container_width=True)

        # Histogram
        fig = go.Figure(data=[go.Histogram(x=err_vals, nbinsx=50,
                                           marker_color="#ff6361")])
        fig.update_layout(title="Disparity Error Distribution",
                          xaxis_title="Absolute Error (px)",
                          yaxis_title="Pixel Count",
                          template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No overlapping valid pixels between SGBM disparity and ground truth.")


# ===================================================================
#  Step 4 — Object Distance from Detections
# ===================================================================
st.divider()
st.subheader("Step 4: Object Distance Estimation")

all_dets = []
all_dets.extend(("RCE", *d) for d in rce_dets)
all_dets.extend(("CNN", *d) for d in cnn_dets)

if not all_dets and crop_bbox is not None:
    st.info("No detections from the Real-Time Detection page. Using the **crop bounding box** as a fallback.")
    x0, y0, x1, y1 = crop_bbox
    all_dets.append(("Crop", x0, y0, x1, y1, "object", 1.0))
elif not all_dets:
    st.warning("No detections found. Run **Real-Time Detection** first, or define a crop in **Data Lab**.")
    st.stop()

if focal <= 0:
    st.warning("Focal length is 0 — cannot compute depth. Upload a valid config in **Data Lab**.")
    st.stop()

rows = []
det_overlay = img_l.copy()

for source, dx1, dy1, dx2, dy2, lbl, conf in all_dets:
    dx1, dy1, dx2, dy2 = int(dx1), int(dy1), int(dx2), int(dy2)

    # Clamp to image bounds
    H, W = depth_map.shape[:2]
    dx1c = max(0, min(dx1, W - 1))
    dy1c = max(0, min(dy1, H - 1))
    dx2c = max(0, min(dx2, W))
    dy2c = max(0, min(dy2, H))

    roi_depth = depth_map[dy1c:dy2c, dx1c:dx2c]
    roi_disp  = disp[dy1c:dy2c, dx1c:dx2c]
    roi_valid = roi_depth[roi_depth > 0]

    if len(roi_valid) > 0:
        med_depth  = float(np.median(roi_valid))
        mean_depth = float(np.mean(roi_valid))
        med_disp   = float(np.median(roi_disp[roi_disp > 0])) if (roi_disp > 0).any() else 0
    else:
        med_depth = mean_depth = med_disp = 0.0

    # Ground truth depth at this region
    gt_depth_val = 0.0
    if gt_left is not None:
        gt_roi = gt_left[dy1c:dy2c, dx1c:dx2c]
        gt_roi_valid = gt_roi[np.isfinite(gt_roi) & (gt_roi > 0)]
        if len(gt_roi_valid) > 0:
            gt_med_disp  = float(np.median(gt_roi_valid))
            gt_depth_val = (focal * baseline) / (gt_med_disp + doffs) if (gt_med_disp + doffs) > 0 else 0

    error_mm = abs(med_depth - gt_depth_val) if gt_depth_val > 0 else float("nan")

    rows.append({
        "Source":        source,
        "Box":           f"({dx1},{dy1})→({dx2},{dy2})",
        "Confidence":    f"{conf:.1%}" if isinstance(conf, float) else str(conf),
        "Med Disparity": f"{med_disp:.1f} px",
        "Med Depth":     f"{med_depth:.0f} mm",
        "Mean Depth":    f"{mean_depth:.0f} mm",
        "GT Depth":      f"{gt_depth_val:.0f} mm" if gt_depth_val > 0 else "N/A",
        "Error":         f"{error_mm:.0f} mm" if not np.isnan(error_mm) else "N/A",
    })

    # Draw on overlay
    color = (0, 255, 0) if "RCE" in source else (0, 0, 255) if "CNN" in source else (255, 255, 0)
    cv2.rectangle(det_overlay, (dx1c, dy1c), (dx2c, dy2c), color, 2)
    depth_str = f"{med_depth / 1000:.2f}m" if med_depth > 0 else "?"
    cv2.putText(det_overlay, f"{source} {depth_str}",
                (dx1c, max(dy1c - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show overlay
st.image(cv2.cvtColor(det_overlay, cv2.COLOR_BGR2RGB),
         caption="Detections with Estimated Distance",
         use_container_width=True)

# Table
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Primary detection summary
if rows:
    best = rows[0]
    st.divider()
    st.subheader("🎯 Primary Detection — Distance")
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Estimated Depth", best["Med Depth"])
    bc2.metric("Ground Truth", best["GT Depth"])
    bc3.metric("Absolute Error", best["Error"])
