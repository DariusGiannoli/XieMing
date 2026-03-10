import streamlit as st
import cv2
import numpy as np
import io

st.set_page_config(page_title="Data Lab", layout="wide")

st.title("🧪 Data Lab: Stereo Asset Loader")
st.write("Upload your stereo images, camera configuration, and ground truth depth maps.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_pfm(file_bytes: bytes) -> np.ndarray:
    """Parse a PFM (Portable Float Map) and return a float32 ndarray."""
    buf = io.BytesIO(file_bytes)
    header = buf.readline().decode("ascii").strip()
    if header not in ("Pf", "PF"):
        raise ValueError(f"Not a valid PFM file (header: {header!r})")
    color = header == "PF"
    line = buf.readline().decode("ascii").strip()
    while line.startswith("#"):
        line = buf.readline().decode("ascii").strip()
    w, h = map(int, line.split())
    scale = float(buf.readline().decode("ascii").strip())
    endian = "<" if scale < 0 else ">"
    channels = 3 if color else 1
    data = np.frombuffer(buf.read(), dtype=np.dtype(endian + "f4"))
    data = data.reshape((h, w, channels) if color else (h, w))
    return np.flipud(data)


def vis_depth(depth: np.ndarray) -> np.ndarray:
    """Normalise depth to [0,1] for display, ignoring non-finite values."""
    finite = depth[np.isfinite(depth)]
    d = np.nan_to_num(depth, nan=0.0, posinf=float(finite.max()))
    return (d / d.max()).astype(np.float32) if d.max() > 0 else d.astype(np.float32)


def augment(img: np.ndarray, brightness: float, contrast: float,
            rotation: float, flip_h: bool, flip_v: bool,
            noise: float, blur: int, shift_x: int, shift_y: int) -> np.ndarray:
    """Apply a chain of augmentations to a BGR crop."""
    out = img.astype(np.float32)

    # Brightness / Contrast:  out = contrast * out + brightness_offset
    out = np.clip(contrast * out + brightness, 0, 255)

    # Gaussian noise
    if noise > 0:
        out = np.clip(out + np.random.normal(0, noise, out.shape), 0, 255)

    out = out.astype(np.uint8)

    # Blur (kernel must be odd)
    k = blur * 2 + 1
    if k > 1:
        out = cv2.GaussianBlur(out, (k, k), 0)

    # Rotation
    if rotation != 0:
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Translation
    if shift_x != 0 or shift_y != 0:
        h, w = out.shape[:2]
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Flips
    if flip_h:
        out = cv2.flip(out, 1)
    if flip_v:
        out = cv2.flip(out, 0)

    return out


# --- Session State Initialization ---
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

if "pipeline_data" not in st.session_state:
    st.session_state["pipeline_data"] = {}

# ---------------------------------------------------------------------------
# Step 1 — Upload Assets
# ---------------------------------------------------------------------------
st.subheader("Step 1: Upload Assets")
col1, col2 = st.columns(2)

with col1:
    up_l = st.file_uploader("Left Image (Reference)", type=["png", "jpg", "jpeg"])
    if up_l:
        if up_l.size > MAX_UPLOAD_BYTES:
            st.error(f"❌ Left image too large ({up_l.size / 1e6:.1f} MB). Max 50 MB.")
            up_l = None
        else:
            img_l_preview = cv2.imdecode(np.frombuffer(up_l.read(), np.uint8), cv2.IMREAD_COLOR)
            up_l.seek(0)
            st.image(cv2.cvtColor(img_l_preview, cv2.COLOR_BGR2RGB), caption="Left Image Preview", use_container_width=True)

    up_conf = st.file_uploader("Camera Config (.txt or .conf)", type=["txt", "conf"])

    up_gt_l = st.file_uploader("Left Ground Truth Depth (.pfm)", type=["pfm"])
    if up_gt_l:
        try:
            gt_l_prev = read_pfm(up_gt_l.read());  up_gt_l.seek(0)
            st.image(vis_depth(gt_l_prev), caption="Left GT Depth Preview", use_container_width=True)
        except (ValueError, Exception) as e:
            st.error(f"❌ Invalid PFM file (left): {e}")
            up_gt_l = None

with col2:
    up_r = st.file_uploader("Right Image (Stereo Match)", type=["png", "jpg", "jpeg"])
    if up_r:
        if up_r.size > MAX_UPLOAD_BYTES:
            st.error(f"❌ Right image too large ({up_r.size / 1e6:.1f} MB). Max 50 MB.")
            up_r = None
        else:
            img_r_preview = cv2.imdecode(np.frombuffer(up_r.read(), np.uint8), cv2.IMREAD_COLOR)
            up_r.seek(0)
            st.image(cv2.cvtColor(img_r_preview, cv2.COLOR_BGR2RGB), caption="Right Image Preview", use_container_width=True)

    up_gt_r = st.file_uploader("Right Ground Truth Depth (.pfm)", type=["pfm"])
    if up_gt_r:
        try:
            gt_r_prev = read_pfm(up_gt_r.read());  up_gt_r.seek(0)
            st.image(vis_depth(gt_r_prev), caption="Right GT Depth Preview", use_container_width=True)
        except (ValueError, Exception) as e:
            st.error(f"❌ Invalid PFM file (right): {e}")
            up_gt_r = None

# ---------------------------------------------------------------------------
# Step 2 — Full pipeline (requires all 5 files)
# ---------------------------------------------------------------------------
if up_l and up_r and up_conf and up_gt_l and up_gt_r:
    img_l       = cv2.imdecode(np.frombuffer(up_l.read(),    np.uint8), cv2.IMREAD_COLOR)
    img_r       = cv2.imdecode(np.frombuffer(up_r.read(),    np.uint8), cv2.IMREAD_COLOR)
    conf_content = up_conf.read().decode("utf-8")
    gt_depth_l  = read_pfm(up_gt_l.read())
    gt_depth_r  = read_pfm(up_gt_r.read())

    st.success("✅ All assets loaded successfully!")

    # --- Stereo pair ---
    st.divider()
    st.subheader("Step 2: Asset Visualization")
    st.write("### 📸 Stereo Pair")
    v1, v2 = st.columns(2)
    v1.image(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB), caption="Left View", use_container_width=True)
    v2.image(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB), caption="Right View", use_container_width=True)

    # --- Ground truth maps ---
    st.write("### 📊 Ground Truth Depth Maps")
    d1, d2 = st.columns(2)
    d1.image(vis_depth(gt_depth_l), caption="Left GT Depth (Normalized)", use_container_width=True)
    d2.image(vis_depth(gt_depth_r), caption="Right GT Depth (Normalized)", use_container_width=True)

    # --- Config ---
    with st.expander("📄 Camera Configuration"):
        st.text_area("Raw Config", conf_content, height=200)

    # -----------------------------------------------------------------------
    # Step 3 — Crop ROI(s) from Left Image  (Multi-Object)
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Step 3: Crop Region(s) of Interest")
    st.write("Define one or more bounding boxes — each becomes a separate class for recognition.")

    H, W = img_l.shape[:2]

    # Manage list of ROIs in session state
    if "rois" not in st.session_state:
        st.session_state["rois"] = [{"label": "object", "x0": 0, "y0": 0,
                                      "x1": min(W, 100), "y1": min(H, 100)}]

    def _add_roi():
        if len(st.session_state["rois"]) >= 20:
            return
        st.session_state["rois"].append(
            {"label": f"object_{len(st.session_state['rois'])+1}",
             "x0": 0, "y0": 0,
             "x1": min(W, 100), "y1": min(H, 100)})

    def _remove_roi(idx):
        if len(st.session_state["rois"]) > 1:
            st.session_state["rois"].pop(idx)

    ROI_COLORS = [(0,255,0), (255,0,0), (0,0,255), (255,255,0),
                  (255,0,255), (0,255,255), (128,255,0), (255,128,0)]

    for i, roi in enumerate(st.session_state["rois"]):
        color = ROI_COLORS[i % len(ROI_COLORS)]
        color_hex = "#{:02x}{:02x}{:02x}".format(*color)
        with st.container(border=True):
            hc1, hc2, hc3 = st.columns([3, 6, 1])
            hc1.markdown(f"**ROI {i+1}** <span style='color:{color_hex}'>■</span>",
                         unsafe_allow_html=True)
            roi["label"] = hc2.text_input("Class Label", roi["label"],
                                           key=f"roi_lbl_{i}")
            if len(st.session_state["rois"]) > 1:
                hc3.button("✕", key=f"roi_del_{i}",
                           on_click=_remove_roi, args=(i,))

            cr1, cr2, cr3, cr4 = st.columns(4)
            roi["x0"] = int(cr1.number_input("X start", 0, W-2, int(roi["x0"]),
                                              step=1, key=f"roi_x0_{i}"))
            roi["y0"] = int(cr2.number_input("Y start", 0, H-2, int(roi["y0"]),
                                              step=1, key=f"roi_y0_{i}"))
            roi["x1"] = int(cr3.number_input("X end", roi["x0"]+1, W,
                                              min(W, int(roi["x1"])),
                                              step=1, key=f"roi_x1_{i}"))
            roi["y1"] = int(cr4.number_input("Y end", roi["y0"]+1, H,
                                              min(H, int(roi["y1"])),
                                              step=1, key=f"roi_y1_{i}"))

    st.button("➕ Add Another ROI", on_click=_add_roi,
              disabled=len(st.session_state["rois"]) >= 20)

    # Draw all ROIs on the image
    overlay = img_l.copy()
    crops = []
    for i, roi in enumerate(st.session_state["rois"]):
        color = ROI_COLORS[i % len(ROI_COLORS)]
        x0, y0, x1, y1 = roi["x0"], roi["y0"], roi["x1"], roi["y1"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        cv2.putText(overlay, roi["label"], (x0, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        crops.append(img_l[y0:y1, x0:x1].copy())

    ov1, ov2 = st.columns([3, 2])
    ov1.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
              caption="Left Image — ROIs highlighted", use_container_width=True)
    with ov2:
        for i, (c, roi) in enumerate(zip(crops, st.session_state["rois"])):
            st.image(cv2.cvtColor(c, cv2.COLOR_BGR2RGB),
                     caption=f"{roi['label']} ({c.shape[1]}×{c.shape[0]})",
                     width=160)

    # For backward compatibility: first ROI is the "primary"
    crop_bgr = crops[0]
    x0, y0, x1, y1 = (st.session_state["rois"][0]["x0"],
                       st.session_state["rois"][0]["y0"],
                       st.session_state["rois"][0]["x1"],
                       st.session_state["rois"][0]["y1"])

    # -----------------------------------------------------------------------
    # Step 4 — Data Augmentation
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Step 4: Data Augmentation")
    st.write("Tune the parameters below — the augmented crop updates live.")

    ac1, ac2 = st.columns(2)
    with ac1:
        brightness = st.slider("Brightness offset",  -100, 100,  0, step=1)
        contrast   = st.slider("Contrast scale",      0.5,  3.0, 1.0, step=0.05)
        rotation   = st.slider("Rotation (°)",        -180, 180,  0, step=1)
        noise      = st.slider("Gaussian noise σ",     0,   50,   0, step=1)
    with ac2:
        blur       = st.slider("Blur kernel (0 = off)", 0,  10,   0, step=1)
        shift_x    = st.slider("Shift X (px)",         -100, 100,  0, step=1)
        shift_y    = st.slider("Shift Y (px)",         -100, 100,  0, step=1)
        flip_h     = st.checkbox("Flip Horizontal")
        flip_v     = st.checkbox("Flip Vertical")

    aug = augment(crop_bgr, brightness, contrast, rotation,
                  flip_h, flip_v, noise, blur, shift_x, shift_y)

    # Apply same augmentation to all crops
    all_augs = [augment(c, brightness, contrast, rotation,
                        flip_h, flip_v, noise, blur, shift_x, shift_y)
                for c in crops]

    aug_col1, aug_col2 = st.columns(2)
    aug_col1.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                   caption="Original Crop (ROI 1)", use_container_width=True)
    aug_col2.image(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB),
                   caption="Augmented Crop (ROI 1)", use_container_width=True)

    if len(crops) > 1:
        st.caption(f"Augmentation applied identically to all {len(crops)} ROIs.")

    # -----------------------------------------------------------------------
    # Step 5 — Lock & Store
    # -----------------------------------------------------------------------
    st.divider()
    if st.button("🚀 Lock Data & Proceed to Benchmark"):
        if not st.session_state.get("rois") or len(st.session_state["rois"]) == 0:
            st.error("❌ Define at least one ROI before locking!")
            st.stop()
        rois_data = []
        for i, roi in enumerate(st.session_state["rois"]):
            rois_data.append({
                "label":    roi["label"],
                "bbox":     (roi["x0"], roi["y0"], roi["x1"], roi["y1"]),
                "crop":     crops[i],
                "crop_aug": all_augs[i],
            })

        st.session_state["pipeline_data"] = {
            "left":       img_l,
            "right":      img_r,
            "gt_left":    gt_depth_l,
            "gt_right":   gt_depth_r,
            "conf_raw":   conf_content,
            # Backward compatibility: first ROI
            "crop":       crop_bgr,
            "crop_aug":   aug,
            "crop_bbox":  (x0, y0, x1, y1),
            # Multi-object
            "rois":       rois_data,
        }
        st.success(f"Data stored with **{len(rois_data)} ROI(s)**! "
                   f"Move to Feature Lab.")

else:
    st.info("Please upload all 5 files (left image, right image, config, left GT, right GT) to proceed.")