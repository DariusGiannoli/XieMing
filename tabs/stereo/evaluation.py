"""Stereo Evaluation — Stage 6 of the Stereo + Depth pipeline."""

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

from src.models import BACKBONES


def _iou(a, b):
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-6)


def match_detections(dets, gt_list, iou_thr):
    dets_sorted = sorted(dets, key=lambda d: d[5], reverse=True)
    matched_gt = set()
    results = []
    for det in dets_sorted:
        det_box = det[:4]
        best_iou, best_gt_idx, best_gt_label = 0.0, -1, None
        for gi, (gt_box, gt_label) in enumerate(gt_list):
            if gi in matched_gt:
                continue
            iou_val = _iou(det_box, gt_box)
            if iou_val > best_iou:
                best_iou, best_gt_idx, best_gt_label = iou_val, gi, gt_label
        if best_iou >= iou_thr and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            results.append((det, best_gt_label, best_iou))
        else:
            results.append((det, None, best_iou))
    return results, len(gt_list) - len(matched_gt), matched_gt


def compute_pr_curve(dets, gt_list, iou_thr, steps=50):
    if not dets:
        return [], [], [], []
    thresholds = np.linspace(0.0, 1.0, steps)
    precisions, recalls, f1s = [], [], []
    for thr in thresholds:
        filtered = [d for d in dets if d[5] >= thr]
        if not filtered:
            precisions.append(1.0); recalls.append(0.0); f1s.append(0.0)
            continue
        matched, n_missed, _ = match_detections(filtered, gt_list, iou_thr)
        tp = sum(1 for _, gt_lbl, _ in matched if gt_lbl is not None)
        fp = sum(1 for _, gt_lbl, _ in matched if gt_lbl is None)
        fn = n_missed
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    return thresholds.tolist(), precisions, recalls, f1s


def build_confusion_matrix(dets, gt_list, iou_thr):
    gt_labels = sorted(set(lbl for _, lbl in gt_list))
    all_labels = gt_labels + ["background"]
    n = len(all_labels)
    matrix = np.zeros((n, n), dtype=int)
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    matched, n_missed, matched_gt_indices = match_detections(dets, gt_list, iou_thr)
    for det, gt_lbl, _ in matched:
        pred_lbl = det[4]
        if gt_lbl is not None:
            pi = label_to_idx.get(pred_lbl, label_to_idx["background"])
            gi = label_to_idx[gt_lbl]
            matrix[pi][gi] += 1
        else:
            pi = label_to_idx.get(pred_lbl, label_to_idx["background"])
            matrix[pi][label_to_idx["background"]] += 1
    for gi, (_, gt_lbl) in enumerate(gt_list):
        if gi not in matched_gt_indices:
            matrix[label_to_idx["background"]][label_to_idx[gt_lbl]] += 1
    return matrix, all_labels


def render():
    st.title("📈 Evaluation: Confusion Matrix & PR Curves")

    pipe = st.session_state.get("stereo_pipeline")
    if not pipe:
        st.error("Complete the **Data Lab** first.")
        st.stop()

    crop     = pipe.get("crop")
    crop_aug = pipe.get("crop_aug", crop)
    bbox     = pipe.get("crop_bbox", (0, 0, crop.shape[1], crop.shape[0])) if crop is not None else None
    rois     = pipe.get("rois", [{"label": "object", "bbox": bbox,
                                   "crop": crop, "crop_aug": crop_aug}])

    rce_dets = pipe.get("rce_dets")
    cnn_dets = pipe.get("cnn_dets")
    orb_dets = pipe.get("orb_dets")

    if rce_dets is None and cnn_dets is None and orb_dets is None:
        st.warning("Run detection first in **Real-Time Detection**.")
        st.stop()

    gt_boxes = [(roi["bbox"], roi["label"]) for roi in rois]

    st.sidebar.subheader("Evaluation Settings")
    iou_thresh = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.5, 0.05,
                                    help="Minimum IoU to count as TP",
                                    key="stereo_eval_iou")

    st.subheader("Ground Truth (from Data Lab ROIs)")
    st.caption(f"{len(gt_boxes)} ground-truth ROIs defined")
    gt_vis = pipe["test_image"].copy()
    for (bx0, by0, bx1, by1), lbl in gt_boxes:
        cv2.rectangle(gt_vis, (bx0, by0), (bx1, by1), (0, 255, 255), 2)
        cv2.putText(gt_vis, lbl, (bx0, by0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    st.image(cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB),
             caption="Ground Truth Annotations", use_container_width=True)
    st.divider()

    methods = {}
    if rce_dets is not None:
        methods["RCE"] = rce_dets
    if cnn_dets is not None:
        methods["CNN"] = cnn_dets
    if orb_dets is not None:
        methods["ORB"] = orb_dets

    # Confusion Matrices
    st.subheader("🔲 Confusion Matrices")
    cm_cols = st.columns(len(methods))
    for col, (name, dets) in zip(cm_cols, methods.items()):
        with col:
            st.markdown(f"**{name}**")
            matrix, labels = build_confusion_matrix(dets, gt_boxes, iou_thresh)
            fig_cm = ff.create_annotated_heatmap(
                z=matrix.tolist(), x=labels, y=labels,
                colorscale="Blues", showscale=True)
            fig_cm.update_layout(title=f"{name} Confusion Matrix",
                                 xaxis_title="Actual", yaxis_title="Predicted",
                                 template="plotly_dark", height=350)
            fig_cm.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_cm, use_container_width=True)

            matched, n_missed, _ = match_detections(dets, gt_boxes, iou_thresh)
            tp = sum(1 for _, g, _ in matched if g is not None)
            fp = sum(1 for _, g, _ in matched if g is None)
            fn = n_missed
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{prec:.1%}")
            m2.metric("Recall", f"{rec:.1%}")
            m3.metric("F1 Score", f"{f1:.1%}")

    # PR Curves
    st.divider()
    st.subheader("📉 Precision-Recall Curves")
    method_colors = {"RCE": "#00ff88", "CNN": "#4488ff", "ORB": "#ff8800"}
    fig_pr = go.Figure()
    fig_f1 = go.Figure()
    summary_rows = []

    for name, dets in methods.items():
        thrs, precs, recs, f1s = compute_pr_curve(dets, gt_boxes, iou_thresh)
        clr = method_colors.get(name, "#ffffff")
        fig_pr.add_trace(go.Scatter(
            x=recs, y=precs, mode="lines+markers",
            name=name, line=dict(color=clr, width=2), marker=dict(size=4)))
        fig_f1.add_trace(go.Scatter(
            x=thrs, y=f1s, mode="lines",
            name=name, line=dict(color=clr, width=2)))
        ap = float(np.trapz(precs, recs)) if recs and precs else 0.0
        best_f1_idx = int(np.argmax(f1s)) if f1s else 0
        summary_rows.append({
            "Method": name,
            "AP": f"{abs(ap):.3f}",
            "Best F1": f"{f1s[best_f1_idx]:.3f}" if f1s else "N/A",
            "@ Threshold": f"{thrs[best_f1_idx]:.2f}" if thrs else "N/A",
            "Detections": len(dets),
        })

    fig_pr.update_layout(title="Precision vs Recall",
                         xaxis_title="Recall", yaxis_title="Precision",
                         template="plotly_dark", height=400,
                         xaxis=dict(range=[0, 1.05]), yaxis=dict(range=[0, 1.05]))
    fig_f1.update_layout(title="F1 Score vs Confidence Threshold",
                         xaxis_title="Confidence Threshold", yaxis_title="F1 Score",
                         template="plotly_dark", height=400,
                         xaxis=dict(range=[0, 1.05]), yaxis=dict(range=[0, 1.05]))
    pc1, pc2 = st.columns(2)
    pc1.plotly_chart(fig_pr, use_container_width=True)
    pc2.plotly_chart(fig_f1, use_container_width=True)

    # Summary Table
    st.divider()
    st.subheader("📊 Summary")
    import pandas as pd
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.caption(f"All metrics computed at IoU threshold = **{iou_thresh:.2f}**.")
