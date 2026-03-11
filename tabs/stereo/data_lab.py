"""Stereo Data Lab — Stage 1 of the Stereo + Depth pipeline."""

import streamlit as st
import cv2
import numpy as np
import os

from utils.middlebury_loader import (
    DEFAULT_MIDDLEBURY_ROOT, scan_dataset_root, load_stereo_pair,
    read_pfm_bytes, parse_calib,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _vis_depth(depth: np.ndarray) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if len(finite) == 0:
        return np.zeros_like(depth, dtype=np.float32)
    d = np.nan_to_num(depth, nan=0.0, posinf=float(finite.max()))
    return (d / (d.max() + 1e-8)).astype(np.float32)


def _augment(img, brightness, contrast, rotation,
             flip_h, flip_v, noise, blur, shift_x, shift_y):
    out = img.astype(np.float32)
    out = np.clip(contrast * out + brightness, 0, 255)
    if noise > 0:
        out = np.clip(out + np.random.normal(0, noise, out.shape), 0, 255)
    out = out.astype(np.uint8)
    k = blur * 2 + 1
    if k > 1:
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rotation != 0:
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if shift_x != 0 or shift_y != 0:
        h, w = out.shape[:2]
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if flip_h:
        out = cv2.flip(out, 1)
    if flip_v:
        out = cv2.flip(out, 0)
    return out


ROI_COLORS = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),
              (255,0,255),(0,255,255),(128,255,0),(255,128,0)]
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def render():
    st.header("🧪 Data Lab — Stereo + Depth")
    st.info("**How this works:** Define your object of interest in the LEFT image. "
            "The system trains on it and attempts to recognise it in the RIGHT "
            "image — a genuinely different viewpoint.")

    source = st.radio("Data source",
                      ["📦 Middlebury Dataset", "📁 Upload your own files"],
                      horizontal=True, key="stereo_source")

    # ===================================================================
    #  Middlebury loader
    # ===================================================================
    if source == "📦 Middlebury Dataset":
        scenes = scan_dataset_root()
        if not scenes:
            st.error("No valid Middlebury scenes found in ./data/middlebury/")
            return

        scene_name = st.selectbox("Select scene", scenes, key="stereo_scene")
        scene_path = os.path.join(DEFAULT_MIDDLEBURY_ROOT, scene_name)
        data = load_stereo_pair(scene_path)

        img_l, img_r = data["left"], data["right"]
        calib = data["calib"]
        gt_disp = data["disparity_gt"]
        conf_raw = calib.get("conf_raw", "")

        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB),
                 caption="🟦 TRAIN IMAGE (Left)", use_container_width=True)
        c2.image(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB),
                 caption="🟥 TEST IMAGE (Right)", use_container_width=True)

        with st.expander("📄 Camera Calibration"):
            st.text_area("calib.txt", conf_raw, height=200, disabled=True)

        if gt_disp is not None:
            st.image(_vis_depth(gt_disp),
                     caption="Ground Truth Disparity (disp0.pfm)",
                     use_container_width=True)

    # ===================================================================
    #  Custom upload
    # ===================================================================
    else:
        uc1, uc2 = st.columns(2)
        with uc1:
            up_l = st.file_uploader("Left Image (TRAIN)", type=["png","jpg","jpeg"],
                                    key="stereo_up_l")
        with uc2:
            up_r = st.file_uploader("Right Image (TEST)", type=["png","jpg","jpeg"],
                                    key="stereo_up_r")

        up_conf = st.file_uploader("Camera Config (.txt) — required for depth",
                                   type=["txt","conf"], key="stereo_up_conf")
        up_pfm = st.file_uploader("Ground Truth Disparity (.pfm) — optional",
                                  type=["pfm"], key="stereo_up_pfm")

        if not (up_l and up_r):
            st.info("Upload at least a left and right image to proceed.")
            return

        if up_l.size > MAX_UPLOAD_BYTES or up_r.size > MAX_UPLOAD_BYTES:
            st.error("Image too large (max 50 MB).")
            return

        img_l = cv2.imdecode(np.frombuffer(up_l.read(), np.uint8), cv2.IMREAD_COLOR); up_l.seek(0)
        img_r = cv2.imdecode(np.frombuffer(up_r.read(), np.uint8), cv2.IMREAD_COLOR); up_r.seek(0)

        calib = {}
        conf_raw = ""
        if up_conf:
            conf_raw = up_conf.read().decode("utf-8"); up_conf.seek(0)
            calib = parse_calib.__wrapped__(conf_raw) if hasattr(parse_calib, '__wrapped__') else {}
            # manual parse since parse_calib expects a file path
            import re
            for line in conf_raw.strip().splitlines():
                line = line.strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if "[" in val:
                    nums = list(map(float, re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)))
                    calib[key] = np.array(nums).reshape(3, 3) if len(nums) == 9 else nums
                else:
                    try:
                        calib[key] = float(val)
                    except ValueError:
                        calib[key] = val
            cam0 = calib.get("cam0")
            if isinstance(cam0, np.ndarray) and cam0.shape == (3, 3):
                calib["fx"] = float(cam0[0, 0])
            calib["conf_raw"] = conf_raw
        else:
            st.warning("⚠️ No calibration file — depth estimation will be disabled.")

        gt_disp = None
        if up_pfm:
            try:
                gt_disp = read_pfm_bytes(up_pfm.read()); up_pfm.seek(0)
            except Exception as e:
                st.error(f"Invalid PFM: {e}")

        scene_name = "custom_upload"

        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB),
                 caption="🟦 TRAIN IMAGE (Left)", use_container_width=True)
        c2.image(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB),
                 caption="🟥 TEST IMAGE (Right)", use_container_width=True)

    # ===================================================================
    #  ROI Definition (on LEFT / train image)
    # ===================================================================
    st.divider()
    st.subheader("Step 2: Crop Region(s) of Interest")
    st.write("Define bounding boxes on the **TRAIN (left) image** — each becomes a separate class.")

    H, W = img_l.shape[:2]

    if "stereo_rois" not in st.session_state:
        st.session_state["stereo_rois"] = [
            {"label": "object", "x0": 0, "y0": 0,
             "x1": min(W, 100), "y1": min(H, 100)}
        ]

    def _add_roi():
        if len(st.session_state["stereo_rois"]) >= 20:
            return
        st.session_state["stereo_rois"].append(
            {"label": f"object_{len(st.session_state['stereo_rois'])+1}",
             "x0": 0, "y0": 0,
             "x1": min(W, 100), "y1": min(H, 100)})

    def _remove_roi(idx):
        if len(st.session_state["stereo_rois"]) > 1:
            st.session_state["stereo_rois"].pop(idx)

    for i, roi in enumerate(st.session_state["stereo_rois"]):
        color = ROI_COLORS[i % len(ROI_COLORS)]
        color_hex = "#{:02x}{:02x}{:02x}".format(*color)
        with st.container(border=True):
            hc1, hc2, hc3 = st.columns([3, 6, 1])
            hc1.markdown(f"**ROI {i+1}** <span style='color:{color_hex}'>■</span>",
                         unsafe_allow_html=True)
            roi["label"] = hc2.text_input("Class Label", roi["label"],
                                           key=f"stereo_roi_lbl_{i}")
            if len(st.session_state["stereo_rois"]) > 1:
                hc3.button("✕", key=f"stereo_roi_del_{i}",
                           on_click=_remove_roi, args=(i,))

            cr1, cr2, cr3, cr4 = st.columns(4)
            roi["x0"] = int(cr1.number_input("X start", 0, W-2, int(roi["x0"]),
                                              step=1, key=f"stereo_roi_x0_{i}"))
            roi["y0"] = int(cr2.number_input("Y start", 0, H-2, int(roi["y0"]),
                                              step=1, key=f"stereo_roi_y0_{i}"))
            roi["x1"] = int(cr3.number_input("X end", roi["x0"]+1, W,
                                              min(W, int(roi["x1"])),
                                              step=1, key=f"stereo_roi_x1_{i}"))
            roi["y1"] = int(cr4.number_input("Y end", roi["y0"]+1, H,
                                              min(H, int(roi["y1"])),
                                              step=1, key=f"stereo_roi_y1_{i}"))

    st.button("➕ Add Another ROI", on_click=_add_roi,
              disabled=len(st.session_state["stereo_rois"]) >= 20,
              key="stereo_add_roi")

    # Draw ROIs
    overlay = img_l.copy()
    crops = []
    for i, roi in enumerate(st.session_state["stereo_rois"]):
        color = ROI_COLORS[i % len(ROI_COLORS)]
        x0, y0, x1, y1 = roi["x0"], roi["y0"], roi["x1"], roi["y1"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        cv2.putText(overlay, roi["label"], (x0, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        crops.append(img_l[y0:y1, x0:x1].copy())

    ov1, ov2 = st.columns([3, 2])
    ov1.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
              caption="TRAIN image — ROIs highlighted", use_container_width=True)
    with ov2:
        for i, (c, roi) in enumerate(zip(crops, st.session_state["stereo_rois"])):
            st.image(cv2.cvtColor(c, cv2.COLOR_BGR2RGB),
                     caption=f"{roi['label']} ({c.shape[1]}×{c.shape[0]})", width=160)

    crop_bgr = crops[0]
    x0, y0, x1, y1 = (st.session_state["stereo_rois"][0]["x0"],
                       st.session_state["stereo_rois"][0]["y0"],
                       st.session_state["stereo_rois"][0]["x1"],
                       st.session_state["stereo_rois"][0]["y1"])

    # ===================================================================
    #  Augmentation
    # ===================================================================
    st.divider()
    st.subheader("Step 3: Data Augmentation")
    ac1, ac2 = st.columns(2)
    with ac1:
        brightness = st.slider("Brightness offset", -100, 100, 0, key="stereo_bright")
        contrast   = st.slider("Contrast scale", 0.5, 3.0, 1.0, 0.05, key="stereo_contrast")
        rotation   = st.slider("Rotation (°)", -180, 180, 0, key="stereo_rot")
        noise      = st.slider("Gaussian noise σ", 0, 50, 0, key="stereo_noise")
    with ac2:
        blur    = st.slider("Blur kernel (0=off)", 0, 10, 0, key="stereo_blur")
        shift_x = st.slider("Shift X (px)", -100, 100, 0, key="stereo_sx")
        shift_y = st.slider("Shift Y (px)", -100, 100, 0, key="stereo_sy")
        flip_h  = st.checkbox("Flip Horizontal", key="stereo_fh")
        flip_v  = st.checkbox("Flip Vertical", key="stereo_fv")

    aug = _augment(crop_bgr, brightness, contrast, rotation,
                   flip_h, flip_v, noise, blur, shift_x, shift_y)
    all_augs = [_augment(c, brightness, contrast, rotation,
                         flip_h, flip_v, noise, blur, shift_x, shift_y)
                for c in crops]

    ag1, ag2 = st.columns(2)
    ag1.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
              caption="Original Crop (ROI 1)", use_container_width=True)
    ag2.image(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB),
              caption="Augmented Crop (ROI 1)", use_container_width=True)

    # ===================================================================
    #  Lock & Store
    # ===================================================================
    st.divider()
    if st.button("🚀 Lock Data & Proceed", key="stereo_lock"):
        rois_data = []
        for i, roi in enumerate(st.session_state["stereo_rois"]):
            rois_data.append({
                "label":    roi["label"],
                "bbox":     (roi["x0"], roi["y0"], roi["x1"], roi["y1"]),
                "crop":     crops[i],
                "crop_aug": all_augs[i],
            })

        st.session_state["stereo_pipeline"] = {
            "train_image":  img_l,
            "test_image":   img_r,
            "calib":        calib if calib else None,
            "disparity_gt": gt_disp,
            "roi":          {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                            "label": st.session_state["stereo_rois"][0]["label"]},
            "crop":         crop_bgr,
            "crop_aug":     aug,
            "crop_bbox":    (x0, y0, x1, y1),
            "rois":         rois_data,
            "source":       "middlebury" if source == "📦 Middlebury Dataset" else "custom",
            "scene_name":   scene_name,
            "conf_raw":     conf_raw,
        }
        st.success(f"✅ Data locked with **{len(rois_data)} ROI(s)**! "
                   f"Proceed to Feature Lab.")
