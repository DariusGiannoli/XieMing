"""Stereo Depth — Stage 7 of the Stereo + Depth pipeline."""

import streamlit as st
import cv2
import numpy as np
import re
import pandas as pd
import plotly.graph_objects as go


def _parse_config(text: str) -> dict:
    params = {}
    if not text or not text.strip():
        return params
    for line in text.strip().splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if "[" in val:
            nums = list(map(float, re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)))
            params[key] = np.array(nums).reshape(3, 3) if len(nums) == 9 else nums
        else:
            try:
                params[key] = float(val)
            except ValueError:
                params[key] = val
    return params


def render():
    st.title("📐 Stereo Geometry: Distance Estimation")

    pipe = st.session_state.get("stereo_pipeline")
    if not pipe or "train_image" not in pipe:
        st.error("Complete **Data Lab** first.")
        st.stop()

    img_l     = pipe["train_image"]
    img_r     = pipe["test_image"]
    gt_left   = pipe.get("disparity_gt")
    conf_raw  = pipe.get("conf_raw", "")
    crop_bbox = pipe.get("crop_bbox")

    rce_dets = pipe.get("rce_dets", [])
    cnn_dets = pipe.get("cnn_dets", [])

    # Parse calibration
    calib_dict = pipe.get("calib")
    if calib_dict is None and conf_raw:
        calib_dict = _parse_config(conf_raw)
    if calib_dict is None:
        calib_dict = {}

    cam0 = calib_dict.get("cam0")
    focal    = float(cam0[0, 0]) if isinstance(cam0, np.ndarray) and cam0.shape == (3, 3) else calib_dict.get("fx", 0.0)
    doffs    = float(calib_dict.get("doffs", 0.0))
    baseline = float(calib_dict.get("baseline", 1.0))
    ndisp    = int(calib_dict.get("ndisp", 128))

    if focal <= 0:
        st.error("Focal length is 0 — the camera config is missing or malformed. "
                 "Depth estimation cannot proceed.")
        st.stop()
    if focal > 10000:
        st.error(f"Focal length ({focal:.0f} px) is suspiciously large.")
        st.stop()
    if baseline <= 0 or baseline > 1000:
        st.error(f"Invalid baseline ({baseline:.1f}).")
        st.stop()

    st.subheader("Camera Calibration")
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Focal Length (px)", f"{focal:.1f}")
    cc2.metric("Baseline (mm)", f"{baseline:.1f}")
    cc3.metric("Doffs (px)", f"{doffs:.2f}")
    cc4.metric("ndisp", str(ndisp))

    with st.expander("Full Calibration"):
        st.json({k: v.tolist() if isinstance(v, np.ndarray) else v
                 for k, v in calib_dict.items() if k != "conf_raw"})
    st.divider()

    if img_l.shape[:2] != img_r.shape[:2]:
        st.error(f"Left ({img_l.shape[1]}×{img_l.shape[0]}) and right "
                 f"({img_r.shape[1]}×{img_r.shape[0]}) images must be the same size.")
        st.stop()

    # Step 1 — Disparity Map
    st.subheader("Step 1: Disparity Map (StereoSGBM)")
    sc1, sc2, sc3 = st.columns(3)
    block_size = sc1.slider("Block Size", 3, 21, 5, step=2, key="stereo_sd_bs")
    p1_mult    = sc2.slider("P1 multiplier", 1, 32, 8, key="stereo_sd_p1")
    p2_mult    = sc3.slider("P2 multiplier", 1, 128, 32, key="stereo_sd_p2")

    @st.cache_data
    def compute_disparity(_left, _right, ndisp, block_size, p1m, p2m):
        gray_l = cv2.cvtColor(_left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(_right, cv2.COLOR_BGR2GRAY)
        nd = max(16, (ndisp // 16) * 16)
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=nd, blockSize=block_size,
            P1=p1m * block_size ** 2, P2=p2m * block_size ** 2,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        return sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0

    with st.spinner("Computing disparity…"):
        try:
            disp = compute_disparity(img_l, img_r, ndisp, block_size, p1_mult, p2_mult)
        except cv2.error as e:
            st.error(f"StereoSGBM failed: {e}")
            st.stop()

    disp_vis = np.clip(disp, 0, None)
    disp_max = disp_vis.max() if disp_vis.max() > 0 else 1.0
    disp_norm = (disp_vis / disp_max * 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_INFERNO)

    dc1, dc2 = st.columns(2)
    dc1.image(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB), caption="Left Image",
              use_container_width=True)
    dc2.image(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB),
              caption="Disparity Map (SGBM)", use_container_width=True)

    # Step 2 — Depth Map
    st.divider()
    st.subheader("Step 2: Depth Map from Disparity")
    st.latex(r"Z = \frac{f \times B}{d + d_{\text{offs}}}")

    valid = (disp + doffs) > 0
    depth_map = np.zeros_like(disp)
    depth_map[valid] = (focal * baseline) / (disp[valid] + doffs)

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

    # Step 3 — Error Analysis
    if gt_left is not None:
        st.divider()
        st.subheader("Step 3: Error Analysis (SGBM vs Ground Truth)")
        gt_disp = gt_left
        if gt_disp.shape[:2] != disp.shape[:2]:
            st.warning("GT shape differs from disparity shape. Resizing GT.")
            gt_disp = cv2.resize(gt_disp, (disp.shape[1], disp.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        gt_valid = np.isfinite(gt_disp) & (gt_disp > 0)
        both_valid = valid & gt_valid
        if both_valid.any():
            disp_err = np.abs(disp - gt_disp)
            disp_err[~both_valid] = 0
            err_vals = disp_err[both_valid]
            mae  = float(np.mean(err_vals))
            rmse = float(np.sqrt(np.mean(err_vals ** 2)))
            bad_2 = float(np.mean(err_vals > 2.0)) * 100

            em1, em2, em3 = st.columns(3)
            em1.metric("MAE (px)", f"{mae:.2f}")
            em2.metric("RMSE (px)", f"{rmse:.2f}")
            em3.metric("Bad-2.0 (%)", f"{bad_2:.1f}%")

            err_clip = np.clip(disp_err, 0, 10)
            err_norm = (err_clip / 10 * 255).astype(np.uint8)
            err_color = cv2.applyColorMap(err_norm, cv2.COLORMAP_HOT)
            st.image(cv2.cvtColor(err_color, cv2.COLOR_BGR2RGB),
                     caption="Disparity Error Map (clipped at 10 px)",
                     use_container_width=True)

            fig = go.Figure(data=[go.Histogram(x=err_vals, nbinsx=50,
                                               marker_color="#ff6361")])
            fig.update_layout(title="Disparity Error Distribution",
                              xaxis_title="Absolute Error (px)",
                              yaxis_title="Pixel Count",
                              template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No overlapping valid pixels.")

    # Step 4 — Object Distance
    st.divider()
    st.subheader("Step 4: Object Distance Estimation")

    all_dets = []
    all_dets.extend(("RCE", *d) for d in rce_dets)
    all_dets.extend(("CNN", *d) for d in cnn_dets)

    if not all_dets and crop_bbox is not None:
        st.info("No detections. Using the **crop bounding box** as a fallback.")
        x0, y0, x1, y1 = crop_bbox
        all_dets.append(("Crop", x0, y0, x1, y1, "object", 1.0))
    elif not all_dets:
        st.warning("No detections found. Run **Real-Time Detection** first.")
        st.stop()

    rows = []
    det_overlay = img_l.copy()

    for source, dx1, dy1, dx2, dy2, lbl, conf in all_dets:
        dx1, dy1, dx2, dy2 = int(dx1), int(dy1), int(dx2), int(dy2)
        H, W = depth_map.shape[:2]
        dx1c, dy1c = max(0, min(dx1, W-1)), max(0, min(dy1, H-1))
        dx2c, dy2c = max(0, min(dx2, W)), max(0, min(dy2, H))

        roi_depth = depth_map[dy1c:dy2c, dx1c:dx2c]
        roi_disp  = disp[dy1c:dy2c, dx1c:dx2c]
        roi_valid = roi_depth[roi_depth > 0]

        if len(roi_valid) > 0:
            med_depth  = float(np.median(roi_valid))
            mean_depth = float(np.mean(roi_valid))
            med_disp   = float(np.median(roi_disp[roi_disp > 0])) if (roi_disp > 0).any() else 0
        else:
            med_depth = mean_depth = med_disp = 0.0

        gt_depth_val = 0.0
        if gt_left is not None:
            gt_roi = gt_left[dy1c:dy2c, dx1c:dx2c]
            gt_roi_valid = gt_roi[np.isfinite(gt_roi) & (gt_roi > 0)]
            if len(gt_roi_valid) > 0:
                gt_med_disp = float(np.median(gt_roi_valid))
                gt_depth_val = (focal * baseline) / (gt_med_disp + doffs) if (gt_med_disp + doffs) > 0 else 0

        error_mm = abs(med_depth - gt_depth_val) if gt_depth_val > 0 else float("nan")

        rows.append({
            "Source": source, "Box": f"({dx1},{dy1})→({dx2},{dy2})",
            "Confidence": f"{conf:.1%}" if isinstance(conf, float) else str(conf),
            "Med Disparity": f"{med_disp:.1f} px",
            "Med Depth": f"{med_depth:.0f} mm", "Mean Depth": f"{mean_depth:.0f} mm",
            "GT Depth": f"{gt_depth_val:.0f} mm" if gt_depth_val > 0 else "N/A",
            "Error": f"{error_mm:.0f} mm" if not np.isnan(error_mm) else "N/A",
        })

        color = (0, 255, 0) if "RCE" in source else (0, 0, 255) if "CNN" in source else (255, 255, 0)
        cv2.rectangle(det_overlay, (dx1c, dy1c), (dx2c, dy2c), color, 2)
        depth_str = f"{med_depth / 1000:.2f}m" if med_depth > 0 else "?"
        cv2.putText(det_overlay, f"{source} {depth_str}",
                    (dx1c, max(dy1c - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    st.image(cv2.cvtColor(det_overlay, cv2.COLOR_BGR2RGB),
             caption="Detections with Estimated Distance", use_container_width=True)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if rows:
        best = rows[0]
        st.divider()
        st.subheader("🎯 Primary Detection — Distance")
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Estimated Depth", best["Med Depth"])
        bc2.metric("Ground Truth", best["GT Depth"])
        bc3.metric("Absolute Error", best["Error"])
