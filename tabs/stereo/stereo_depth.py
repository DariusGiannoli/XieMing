"""Stereo Depth — Stage 7 of the Stereo + Depth pipeline."""

import streamlit as st
import cv2
import numpy as np
import re
import pandas as pd
import plotly.graph_objects as go

from src.depth_nn import load_depth_anything, predict_depth, align_to_gt


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

    # Step 3 — Neural Network Depth (Depth Anything V2)
    st.divider()
    st.subheader("Step 3: Neural Network Depth (Depth Anything V2 — Small)")
    st.caption("Monocular depth estimation on the **left image only** — "
               "no stereo pair needed.  Output is aligned to GT disparity "
               "via least-squares so metrics are directly comparable with SGBM.")

    run_nn = st.checkbox("Run Depth Anything V2", value=False,
                         key="stereo_sd_run_nn")

    nn_disp_aligned = None
    if run_nn:
        with st.spinner("Running Depth Anything V2 (CPU) …"):
            nn_raw, nn_ms = predict_depth(img_l)
        st.success(f"Inference finished in **{nn_ms:.0f} ms**")

        # Align to GT if available, otherwise show raw output
        if gt_left is not None:
            nn_disp_aligned = align_to_gt(nn_raw, gt_left)
        else:
            nn_disp_aligned = nn_raw

        # Visualise
        nn_vis = np.clip(nn_disp_aligned, 0, None)
        nn_max = nn_vis.max() if nn_vis.max() > 0 else 1.0
        nn_norm = (nn_vis / nn_max * 255).astype(np.uint8)
        nn_color = cv2.applyColorMap(nn_norm, cv2.COLORMAP_INFERNO)

        nc1, nc2 = st.columns(2)
        nc1.image(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB),
                  caption="SGBM Disparity", use_container_width=True)
        nc2.image(cv2.cvtColor(nn_color, cv2.COLOR_BGR2RGB),
                  caption="Depth Anything V2 (aligned)", use_container_width=True)

    # Step 4 — Comparative Error Analysis
    if gt_left is not None:
        st.divider()
        st.subheader("Step 4: Error Analysis — SGBM vs Depth Anything vs GT")

        gt_disp = gt_left
        if gt_disp.shape[:2] != disp.shape[:2]:
            st.warning("GT shape differs from disparity shape. Resizing GT.")
            gt_disp = cv2.resize(gt_disp, (disp.shape[1], disp.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        gt_valid = np.isfinite(gt_disp) & (gt_disp > 0)

        # --- helper to compute metrics for any disparity map ---
        def _error_metrics(pred, gt_d, gt_v, valid_pred):
            both = valid_pred & gt_v
            if not both.any():
                return None, None, None, None
            err = np.abs(pred - gt_d)
            err[~both] = 0
            vals = err[both]
            return (
                float(np.mean(vals)),
                float(np.sqrt(np.mean(vals ** 2))),
                float(np.mean(vals > 2.0)) * 100,
                err,
            )

        sgbm_mae, sgbm_rmse, sgbm_bad2, sgbm_err_map = _error_metrics(
            disp, gt_disp, gt_valid, valid)

        nn_mae = nn_rmse = nn_bad2 = nn_err_map = None
        if nn_disp_aligned is not None:
            nn_valid = nn_disp_aligned > 0
            nn_mae, nn_rmse, nn_bad2, nn_err_map = _error_metrics(
                nn_disp_aligned, gt_disp, gt_valid, nn_valid)

        # --- Metric cards ---
        st.markdown("##### Disparity Error Metrics")
        if nn_mae is not None:
            col_hdr, col_sgbm, col_nn = st.columns(3)
            col_hdr.markdown("**Metric**")
            col_sgbm.markdown("**SGBM**")
            col_nn.markdown("**Depth Anything**")

            col_hdr2, col_sgbm2, col_nn2 = st.columns(3)
            col_hdr2.write("MAE (px)")
            col_sgbm2.write(f"{sgbm_mae:.2f}" if sgbm_mae else "N/A")
            col_nn2.write(f"{nn_mae:.2f}")

            col_hdr3, col_sgbm3, col_nn3 = st.columns(3)
            col_hdr3.write("RMSE (px)")
            col_sgbm3.write(f"{sgbm_rmse:.2f}" if sgbm_rmse else "N/A")
            col_nn3.write(f"{nn_rmse:.2f}")

            col_hdr4, col_sgbm4, col_nn4 = st.columns(3)
            col_hdr4.write("Bad-2.0 (%)")
            col_sgbm4.write(f"{sgbm_bad2:.1f}%" if sgbm_bad2 else "N/A")
            col_nn4.write(f"{nn_bad2:.1f}%")
        elif sgbm_mae is not None:
            em1, em2, em3 = st.columns(3)
            em1.metric("MAE (px)", f"{sgbm_mae:.2f}")
            em2.metric("RMSE (px)", f"{sgbm_rmse:.2f}")
            em3.metric("Bad-2.0 (%)", f"{sgbm_bad2:.1f}%")

        # --- Error maps side-by-side ---
        def _err_heatmap(err_map):
            ec = np.clip(err_map, 0, 10)
            en = (ec / 10 * 255).astype(np.uint8)
            return cv2.applyColorMap(en, cv2.COLORMAP_HOT)

        if sgbm_err_map is not None and nn_err_map is not None:
            emc1, emc2 = st.columns(2)
            emc1.image(cv2.cvtColor(_err_heatmap(sgbm_err_map), cv2.COLOR_BGR2RGB),
                       caption="SGBM Error (clipped 10 px)",
                       use_container_width=True)
            emc2.image(cv2.cvtColor(_err_heatmap(nn_err_map), cv2.COLOR_BGR2RGB),
                       caption="Depth Anything Error (clipped 10 px)",
                       use_container_width=True)
        elif sgbm_err_map is not None:
            st.image(cv2.cvtColor(_err_heatmap(sgbm_err_map), cv2.COLOR_BGR2RGB),
                     caption="SGBM Error (clipped 10 px)",
                     use_container_width=True)

        # --- Overlaid histogram ---
        if sgbm_err_map is not None:
            both_sgbm = valid & gt_valid
            sgbm_vals = sgbm_err_map[both_sgbm] if both_sgbm.any() else np.array([])
            traces = [go.Histogram(x=sgbm_vals, nbinsx=50, name="SGBM",
                                   marker_color="#ff6361", opacity=0.7)]
            if nn_err_map is not None:
                nn_v = (nn_disp_aligned > 0) & gt_valid
                nn_vals = nn_err_map[nn_v] if nn_v.any() else np.array([])
                traces.append(go.Histogram(x=nn_vals, nbinsx=50,
                                           name="Depth Anything",
                                           marker_color="#58a6ff",
                                           opacity=0.7))
            fig = go.Figure(data=traces)
            fig.update_layout(barmode="overlay",
                              title="Disparity Error Distribution",
                              xaxis_title="Absolute Error (px)",
                              yaxis_title="Pixel Count",
                              template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

    # Step 5 — Object Distance
    st.divider()
    st.subheader("Step 5: Object Distance Estimation")

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

    # Build NN depth map (metric) from aligned disparity if available
    nn_depth_map = None
    if nn_disp_aligned is not None:
        nn_valid_d = (nn_disp_aligned + doffs) > 0
        nn_depth_map = np.zeros_like(nn_disp_aligned)
        nn_depth_map[nn_valid_d] = (focal * baseline) / (nn_disp_aligned[nn_valid_d] + doffs)

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

        # NN depth for this detection ROI
        nn_med_depth = 0.0
        if nn_depth_map is not None:
            nn_roi = nn_depth_map[dy1c:dy2c, dx1c:dx2c]
            nn_roi_valid = nn_roi[nn_roi > 0]
            if len(nn_roi_valid) > 0:
                nn_med_depth = float(np.median(nn_roi_valid))

        gt_depth_val = 0.0
        if gt_left is not None:
            gt_roi = gt_left[dy1c:dy2c, dx1c:dx2c]
            gt_roi_valid = gt_roi[np.isfinite(gt_roi) & (gt_roi > 0)]
            if len(gt_roi_valid) > 0:
                gt_med_disp = float(np.median(gt_roi_valid))
                gt_depth_val = (focal * baseline) / (gt_med_disp + doffs) if (gt_med_disp + doffs) > 0 else 0

        sgbm_err = abs(med_depth - gt_depth_val) if gt_depth_val > 0 else float("nan")
        nn_err   = abs(nn_med_depth - gt_depth_val) if (gt_depth_val > 0 and nn_med_depth > 0) else float("nan")

        row = {
            "Source": source, "Box": f"({dx1},{dy1})→({dx2},{dy2})",
            "Confidence": f"{conf:.1%}" if isinstance(conf, float) else str(conf),
            "Med Disparity": f"{med_disp:.1f} px",
            "SGBM Depth": f"{med_depth:.0f} mm",
            "GT Depth": f"{gt_depth_val:.0f} mm" if gt_depth_val > 0 else "N/A",
            "SGBM Error": f"{sgbm_err:.0f} mm" if not np.isnan(sgbm_err) else "N/A",
        }
        if nn_depth_map is not None:
            row["NN Depth"] = f"{nn_med_depth:.0f} mm" if nn_med_depth > 0 else "N/A"
            row["NN Error"] = f"{nn_err:.0f} mm" if not np.isnan(nn_err) else "N/A"
        rows.append(row)

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
        bc1.metric("SGBM Depth", best["SGBM Depth"])
        bc2.metric("Ground Truth", best["GT Depth"])
        bc3.metric("SGBM Error", best["SGBM Error"])
        if "NN Depth" in best:
            nc1, nc2, nc3 = st.columns(3)
            nc1.metric("NN Depth", best["NN Depth"])
            nc2.metric("Ground Truth", best["GT Depth"])
            nc3.metric("NN Error", best["NN Error"])
