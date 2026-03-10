"""Stereo Localization Lab — Stage 4 of the Stereo + Depth pipeline."""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.detectors.rce.features import REGISTRY
from src.models import BACKBONES
from src.utils import build_rce_vector
from src.localization import (
    exhaustive_sliding_window,
    image_pyramid,
    coarse_to_fine,
    contour_proposals,
    template_matching,
    STRATEGIES,
)


def render():
    st.title("🔍 Localization Lab")
    st.markdown(
        "Compare **localization strategies** — algorithms that decide *where* "
        "to look in the image.  The recognition head stays the same; only the "
        "search method changes."
    )

    pipe = st.session_state.get("stereo_pipeline")
    if not pipe or "crop" not in pipe:
        st.error("Complete **Data Lab** first (upload assets & define a crop).")
        st.stop()

    test_img  = pipe["test_image"]
    crop      = pipe["crop"]
    crop_aug  = pipe.get("crop_aug", crop)
    bbox      = pipe.get("crop_bbox", (0, 0, crop.shape[1], crop.shape[0]))
    active_mods = pipe.get("active_modules", {k: True for k in REGISTRY})

    x0, y0, x1, y1 = bbox
    win_h, win_w = y1 - y0, x1 - x0

    if win_h <= 0 or win_w <= 0:
        st.error("Invalid window size from crop bbox. "
                 "Go back to **Data Lab** and redefine the ROI.")
        st.stop()

    rce_head    = pipe.get("rce_head")
    has_any_cnn = any(f"cnn_head_{n}" in pipe for n in BACKBONES)

    if rce_head is None and not has_any_cnn:
        st.warning("No trained heads found.  Go to **Model Tuning** first.")
        st.stop()

    def rce_feature_fn(patch_bgr):
        return build_rce_vector(patch_bgr, active_mods)

    # Algorithm Reference
    st.divider()
    with st.expander("📚 **Algorithm Reference** — click to expand", expanded=False):
        tabs = st.tabs([f"{v['icon']} {k}" for k, v in STRATEGIES.items()])
        for tab, (name, meta) in zip(tabs, STRATEGIES.items()):
            with tab:
                st.markdown(f"### {meta['icon']}  {name}")
                st.caption(meta["short"])
                st.markdown(meta["detail"])

    # Configuration
    st.divider()
    st.header("⚙️ Configuration")

    col_head, col_info = st.columns([2, 3])
    with col_head:
        head_options = []
        if rce_head is not None:
            head_options.append("RCE")
        trained_cnns = [n for n in BACKBONES if f"cnn_head_{n}" in pipe]
        head_options.extend(trained_cnns)
        selected_head = st.selectbox("Recognition Head", head_options,
                                      key="stereo_loc_head")

    if selected_head == "RCE":
        feature_fn = rce_feature_fn
        head = rce_head
    else:
        bmeta    = BACKBONES[selected_head]
        backbone = bmeta["loader"]()
        feature_fn = backbone.get_features
        head = pipe[f"cnn_head_{selected_head}"]

    with col_info:
        if selected_head == "RCE":
            mods = [REGISTRY[k]["label"] for k in active_mods if active_mods[k]]
            st.info(f"**RCE** — Modules: {', '.join(mods)}")
        else:
            st.info(f"**{selected_head}** — "
                    f"{BACKBONES[selected_head]['dim']}D feature vector")

    # Algorithm checkboxes
    st.subheader("Select Algorithms to Compare")
    algo_cols = st.columns(5)
    algo_names = list(STRATEGIES.keys())
    algo_checks = {}
    for col, name in zip(algo_cols, algo_names):
        algo_checks[name] = col.checkbox(
            f"{STRATEGIES[name]['icon']} {name}",
            value=(name != "Template Matching"),
            key=f"stereo_chk_{name}")

    any_selected = any(algo_checks.values())

    # Parameters
    st.subheader("Parameters")
    sp1, sp2, sp3 = st.columns(3)
    stride      = sp1.slider("Base Stride (px)", 4, max(win_w, win_h),
                              max(win_w // 4, 4), step=2, key="stereo_loc_stride")
    conf_thresh = sp2.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.05,
                              key="stereo_loc_conf")
    nms_iou     = sp3.slider("NMS IoU Threshold", 0.1, 0.9, 0.3, 0.05,
                              key="stereo_loc_nms")

    with st.expander("🔧 Per-Algorithm Settings"):
        pa1, pa2, pa3 = st.columns(3)
        with pa1:
            st.markdown("**Image Pyramid**")
            pyr_min = st.slider("Min Scale", 0.3, 1.0, 0.5, 0.05, key="stereo_pyr_min")
            pyr_max = st.slider("Max Scale", 1.0, 2.0, 1.5, 0.1, key="stereo_pyr_max")
            pyr_n = st.slider("Number of Scales", 3, 7, 5, key="stereo_pyr_n")
        with pa2:
            st.markdown("**Coarse-to-Fine**")
            c2f_factor = st.slider("Coarse Factor", 2, 8, 4, key="stereo_c2f_factor")
            c2f_radius = st.slider("Refine Radius (strides)", 1, 5, 2, key="stereo_c2f_radius")
        with pa3:
            st.markdown("**Contour Proposals**")
            cnt_low = st.slider("Canny Low", 10, 100, 50, key="stereo_cnt_low")
            cnt_high = st.slider("Canny High", 50, 300, 150, key="stereo_cnt_high")
            cnt_tol = st.slider("Area Tolerance", 1.5, 10.0, 3.0, 0.5, key="stereo_cnt_tol")

    st.caption(
        f"Window: **{win_w}×{win_h} px**  ·  "
        f"Image: **{test_img.shape[1]}×{test_img.shape[0]} px**  ·  "
        f"Stride: **{stride} px**"
    )

    # Run
    st.divider()
    run_btn = st.button("▶  Run Comparison", type="primary",
                         disabled=not any_selected, use_container_width=True,
                         key="stereo_loc_run")

    if run_btn:
        selected_algos = [n for n in algo_names if algo_checks[n]]
        progress = st.progress(0, text="Starting…")
        results = {}
        edge_maps = {}

        for i, name in enumerate(selected_algos):
            progress.progress(i / len(selected_algos), text=f"Running **{name}**…")

            if name == "Exhaustive Sliding Window":
                dets, n, ms, hmap = exhaustive_sliding_window(
                    test_img, win_h, win_w, feature_fn, head,
                    stride, conf_thresh, nms_iou)
            elif name == "Image Pyramid":
                scales = np.linspace(pyr_min, pyr_max, pyr_n).tolist()
                dets, n, ms, hmap = image_pyramid(
                    test_img, win_h, win_w, feature_fn, head,
                    stride, conf_thresh, nms_iou, scales=scales)
            elif name == "Coarse-to-Fine":
                dets, n, ms, hmap = coarse_to_fine(
                    test_img, win_h, win_w, feature_fn, head,
                    stride, conf_thresh, nms_iou,
                    coarse_factor=c2f_factor, refine_radius=c2f_radius)
            elif name == "Contour Proposals":
                dets, n, ms, hmap, edges = contour_proposals(
                    test_img, win_h, win_w, feature_fn, head,
                    conf_thresh, nms_iou,
                    canny_low=cnt_low, canny_high=cnt_high,
                    area_tolerance=cnt_tol)
                edge_maps[name] = edges
            elif name == "Template Matching":
                dets, n, ms, hmap = template_matching(
                    test_img, crop_aug, conf_thresh, nms_iou)

            results[name] = {"dets": dets, "n_proposals": n,
                             "time_ms": ms, "heatmap": hmap}

        progress.progress(1.0, text="Done!")

        # Summary Table
        st.header("📊 Results")
        baseline_ms = results.get("Exhaustive Sliding Window", {}).get("time_ms")
        rows = []
        for name, r in results.items():
            speedup = (baseline_ms / r["time_ms"]
                       if baseline_ms and r["time_ms"] > 0 else None)
            rows.append({
                "Algorithm":   name,
                "Proposals":   r["n_proposals"],
                "Time (ms)":   round(r["time_ms"], 1),
                "Detections":  len(r["dets"]),
                "ms / Proposal": round(r["time_ms"] / max(r["n_proposals"], 1), 4),
                "Speedup": f"{speedup:.1f}×" if speedup else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Detection Images & Heatmaps
        st.subheader("Detection Results")
        COLORS = {
            "Exhaustive Sliding Window": (0, 255, 0),
            "Image Pyramid":             (255, 128, 0),
            "Coarse-to-Fine":            (0, 128, 255),
            "Contour Proposals":         (255, 0, 255),
            "Template Matching":         (0, 255, 255),
        }

        result_tabs = st.tabs(
            [f"{STRATEGIES[n]['icon']} {n}" for n in results])

        for tab, (name, r) in zip(result_tabs, results.items()):
            with tab:
                c1, c2 = st.columns(2)
                color = COLORS.get(name, (0, 255, 0))

                vis = test_img.copy()
                for x1d, y1d, x2d, y2d, _, cf in r["dets"]:
                    cv2.rectangle(vis, (x1d, y1d), (x2d, y2d), color, 2)
                    cv2.putText(vis, f"{cf:.0%}", (x1d, y1d - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                c1.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                         caption=f"{name} — {len(r['dets'])} detections",
                         use_container_width=True)

                hmap = r["heatmap"]
                if hmap.max() > 0:
                    hmap_color = cv2.applyColorMap(
                        (hmap / hmap.max() * 255).astype(np.uint8),
                        cv2.COLORMAP_JET)
                    blend = cv2.addWeighted(test_img, 0.5, hmap_color, 0.5, 0)
                    c2.image(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB),
                             caption=f"{name} — Confidence Heatmap",
                             use_container_width=True)
                else:
                    c2.info("No positive responses above threshold.")

                if name in edge_maps:
                    st.image(edge_maps[name],
                             caption="Canny Edge Map",
                             use_container_width=True, clamp=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Proposals", r["n_proposals"])
                m2.metric("Time", f"{r['time_ms']:.0f} ms")
                m3.metric("Detections", len(r["dets"]))
                m4.metric("ms / Proposal",
                          f"{r['time_ms'] / max(r['n_proposals'], 1):.3f}")

                if r["dets"]:
                    df = pd.DataFrame(r["dets"],
                                      columns=["x1","y1","x2","y2","label","conf"])
                    st.dataframe(df, use_container_width=True, hide_index=True)

        # Performance Charts
        st.subheader("📈 Performance Comparison")
        ch1, ch2 = st.columns(2)
        names  = list(results.keys())
        times  = [results[n]["time_ms"] for n in names]
        props  = [results[n]["n_proposals"] for n in names]
        n_dets = [len(results[n]["dets"]) for n in names]
        colors_hex = ["#00cc66", "#ff8800", "#0088ff", "#ff00ff", "#00cccc"]

        with ch1:
            fig = go.Figure(go.Bar(
                x=names, y=times,
                text=[f"{t:.0f}" for t in times], textposition="auto",
                marker_color=colors_hex[:len(names)]))
            fig.update_layout(title="Total Time (ms)", yaxis_title="ms", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            fig = go.Figure(go.Bar(
                x=names, y=props,
                text=[str(p) for p in props], textposition="auto",
                marker_color=colors_hex[:len(names)]))
            fig.update_layout(title="Proposals Evaluated",
                              yaxis_title="Count", height=400)
            st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        for i, name in enumerate(names):
            fig.add_trace(go.Scatter(
                x=[props[i]], y=[times[i]],
                mode="markers+text",
                marker=dict(size=max(n_dets[i] * 12, 18),
                            color=colors_hex[i % len(colors_hex)]),
                text=[name], textposition="top center", name=name))
        fig.update_layout(
            title="Proposals vs Time  (marker size ∝ detections)",
            xaxis_title="Proposals Evaluated",
            yaxis_title="Time (ms)", height=500)
        st.plotly_chart(fig, use_container_width=True)
