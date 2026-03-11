"""Generalisation Data Lab — Stage 1 of the Generalisation pipeline."""

import streamlit as st
import cv2
import numpy as np
import os

from utils.middlebury_loader import (
    DEFAULT_MIDDLEBURY_ROOT, get_scene_groups, load_single_view,
    read_pfm_bytes,
)


# ------------------------------------------------------------------
# Helpers (shared with stereo data lab)
# ------------------------------------------------------------------

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
    st.header("🧪 Data Lab — Generalisation")
    st.info("**How this works:** Train on one image, test on a completely "
            "different image of the same object. No stereo geometry — "
            "pure recognition generalisation.")

    source = st.radio("Data source",
                      ["📦 Middlebury Multi-View", "📁 Upload your own files"],
                      horizontal=True, key="gen_source")

    # ===================================================================
    #  Middlebury multi-view
    # ===================================================================
    if source == "📦 Middlebury Multi-View":
        groups = get_scene_groups()
        if not groups:
            st.error("No valid Middlebury scenes found in ./data/middlebury/")
            return

        group_name = st.selectbox("Scene group", list(groups.keys()), key="gen_group")
        variants = groups[group_name]

        gc1, gc2 = st.columns(2)
        train_scene = gc1.selectbox("Training scene", variants, key="gen_train_scene")
        available_test = [v for v in variants if v != train_scene]
        if not available_test:
            st.error("Need at least 2 variants in a group.")
            return
        test_scene = gc2.selectbox("Test scene", available_test, key="gen_test_scene")

        train_path = os.path.join(DEFAULT_MIDDLEBURY_ROOT, train_scene)
        test_path  = os.path.join(DEFAULT_MIDDLEBURY_ROOT, test_scene)

        img_train = load_single_view(train_path)
        img_test  = load_single_view(test_path)

        st.markdown("*Both images show the same scene type captured under different "
                    "conditions. The model trains on one variant and must recognise "
                    "the same object class in the other — testing genuine appearance "
                    "generalisation.*")

        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img_train, cv2.COLOR_BGR2RGB),
                 caption=f"🟦 TRAIN IMAGE ({train_scene})", use_container_width=True)
        c2.image(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB),
                 caption=f"🟥 TEST IMAGE ({test_scene})", use_container_width=True)

        scene_group = group_name

    # ===================================================================
    #  Custom upload
    # ===================================================================
    else:
        uc1, uc2 = st.columns(2)
        with uc1:
            up_train = st.file_uploader("Train Image", type=["png","jpg","jpeg"],
                                        key="gen_up_train")
        with uc2:
            up_test = st.file_uploader("Test Image", type=["png","jpg","jpeg"],
                                       key="gen_up_test")

        if not (up_train and up_test):
            st.info("Upload a train and test image to proceed.")
            return

        if up_train.size > MAX_UPLOAD_BYTES or up_test.size > MAX_UPLOAD_BYTES:
            st.error("Image too large (max 50 MB).")
            return

        img_train = cv2.imdecode(np.frombuffer(up_train.read(), np.uint8), cv2.IMREAD_COLOR); up_train.seek(0)
        img_test  = cv2.imdecode(np.frombuffer(up_test.read(), np.uint8), cv2.IMREAD_COLOR); up_test.seek(0)

        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img_train, cv2.COLOR_BGR2RGB),
                 caption="🟦 TRAIN IMAGE", use_container_width=True)
        c2.image(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB),
                 caption="🟥 TEST IMAGE", use_container_width=True)

        train_scene = "custom_train"
        test_scene  = "custom_test"
        scene_group = "custom"

    # ===================================================================
    #  ROI Definition  (on TRAIN image)
    # ===================================================================
    st.divider()
    st.subheader("Step 2: Crop Region(s) of Interest")
    st.write("Define bounding boxes on the **TRAIN image**.")

    H, W = img_train.shape[:2]

    if "gen_rois" not in st.session_state:
        st.session_state["gen_rois"] = [
            {"label": "object", "x0": 0, "y0": 0,
             "x1": min(W, 100), "y1": min(H, 100)}
        ]

    def _add_roi():
        if len(st.session_state["gen_rois"]) >= 20:
            return
        st.session_state["gen_rois"].append(
            {"label": f"object_{len(st.session_state['gen_rois'])+1}",
             "x0": 0, "y0": 0,
             "x1": min(W, 100), "y1": min(H, 100)})

    def _remove_roi(idx):
        if len(st.session_state["gen_rois"]) > 1:
            st.session_state["gen_rois"].pop(idx)

    for i, roi in enumerate(st.session_state["gen_rois"]):
        color = ROI_COLORS[i % len(ROI_COLORS)]
        color_hex = "#{:02x}{:02x}{:02x}".format(*color)
        with st.container(border=True):
            hc1, hc2, hc3 = st.columns([3, 6, 1])
            hc1.markdown(f"**ROI {i+1}** <span style='color:{color_hex}'>■</span>",
                         unsafe_allow_html=True)
            roi["label"] = hc2.text_input("Class Label", roi["label"],
                                           key=f"gen_roi_lbl_{i}")
            if len(st.session_state["gen_rois"]) > 1:
                hc3.button("✕", key=f"gen_roi_del_{i}",
                           on_click=_remove_roi, args=(i,))

            cr1, cr2, cr3, cr4 = st.columns(4)
            roi["x0"] = int(cr1.number_input("X start", 0, W-2, int(roi["x0"]),
                                              step=1, key=f"gen_roi_x0_{i}"))
            roi["y0"] = int(cr2.number_input("Y start", 0, H-2, int(roi["y0"]),
                                              step=1, key=f"gen_roi_y0_{i}"))
            roi["x1"] = int(cr3.number_input("X end", roi["x0"]+1, W,
                                              min(W, int(roi["x1"])),
                                              step=1, key=f"gen_roi_x1_{i}"))
            roi["y1"] = int(cr4.number_input("Y end", roi["y0"]+1, H,
                                              min(H, int(roi["y1"])),
                                              step=1, key=f"gen_roi_y1_{i}"))

    st.button("➕ Add Another ROI", on_click=_add_roi,
              disabled=len(st.session_state["gen_rois"]) >= 20,
              key="gen_add_roi")

    overlay = img_train.copy()
    crops = []
    for i, roi in enumerate(st.session_state["gen_rois"]):
        color = ROI_COLORS[i % len(ROI_COLORS)]
        x0, y0, x1, y1 = roi["x0"], roi["y0"], roi["x1"], roi["y1"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        cv2.putText(overlay, roi["label"], (x0, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        crops.append(img_train[y0:y1, x0:x1].copy())

    ov1, ov2 = st.columns([3, 2])
    ov1.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
              caption="TRAIN image — ROIs highlighted", use_container_width=True)
    with ov2:
        for i, (c, roi) in enumerate(zip(crops, st.session_state["gen_rois"])):
            st.image(cv2.cvtColor(c, cv2.COLOR_BGR2RGB),
                     caption=f"{roi['label']} ({c.shape[1]}×{c.shape[0]})", width=160)

    crop_bgr = crops[0]
    x0 = st.session_state["gen_rois"][0]["x0"]
    y0 = st.session_state["gen_rois"][0]["y0"]
    x1 = st.session_state["gen_rois"][0]["x1"]
    y1 = st.session_state["gen_rois"][0]["y1"]

    # ===================================================================
    #  Augmentation
    # ===================================================================
    st.divider()
    st.subheader("Step 3: Data Augmentation")
    ac1, ac2 = st.columns(2)
    with ac1:
        brightness = st.slider("Brightness offset", -100, 100, 0, key="gen_bright")
        contrast   = st.slider("Contrast scale", 0.5, 3.0, 1.0, 0.05, key="gen_contrast")
        rotation   = st.slider("Rotation (°)", -180, 180, 0, key="gen_rot")
        noise      = st.slider("Gaussian noise σ", 0, 50, 0, key="gen_noise")
    with ac2:
        blur    = st.slider("Blur kernel (0=off)", 0, 10, 0, key="gen_blur")
        shift_x = st.slider("Shift X (px)", -100, 100, 0, key="gen_sx")
        shift_y = st.slider("Shift Y (px)", -100, 100, 0, key="gen_sy")
        flip_h  = st.checkbox("Flip Horizontal", key="gen_fh")
        flip_v  = st.checkbox("Flip Vertical", key="gen_fv")

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
    if st.button("🚀 Lock Data & Proceed", key="gen_lock"):
        rois_data = []
        for i, roi in enumerate(st.session_state["gen_rois"]):
            rois_data.append({
                "label":    roi["label"],
                "bbox":     (roi["x0"], roi["y0"], roi["x1"], roi["y1"]),
                "crop":     crops[i],
                "crop_aug": all_augs[i],
            })

        st.session_state["gen_pipeline"] = {
            "train_image":  img_train,
            "test_image":   img_test,
            "roi":          {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                            "label": st.session_state["gen_rois"][0]["label"]},
            "crop":         crop_bgr,
            "crop_aug":     aug,
            "crop_bbox":    (x0, y0, x1, y1),
            "rois":         rois_data,
            "source":       "middlebury" if source == "📦 Middlebury Multi-View" else "custom",
            "scene_group":  scene_group,
            "train_scene":  train_scene,
            "test_scene":   test_scene,
        }
        st.success(f"✅ Data locked with **{len(rois_data)} ROI(s)**! "
                   f"Proceed to Feature Lab.")
