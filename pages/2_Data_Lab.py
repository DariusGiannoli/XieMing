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
        img_l_preview = cv2.imdecode(np.frombuffer(up_l.read(), np.uint8), cv2.IMREAD_COLOR)
        up_l.seek(0)
        st.image(cv2.cvtColor(img_l_preview, cv2.COLOR_BGR2RGB), caption="Left Image Preview", use_container_width=True)

    up_conf = st.file_uploader("Camera Config (.txt or .conf)", type=["txt", "conf"])

    up_gt_l = st.file_uploader("Left Ground Truth Depth (.pfm)", type=["pfm"])
    if up_gt_l:
        gt_l_prev = read_pfm(up_gt_l.read());  up_gt_l.seek(0)
        st.image(vis_depth(gt_l_prev), caption="Left GT Depth Preview", use_container_width=True)

with col2:
    up_r = st.file_uploader("Right Image (Stereo Match)", type=["png", "jpg", "jpeg"])
    if up_r:
        img_r_preview = cv2.imdecode(np.frombuffer(up_r.read(), np.uint8), cv2.IMREAD_COLOR)
        up_r.seek(0)
        st.image(cv2.cvtColor(img_r_preview, cv2.COLOR_BGR2RGB), caption="Right Image Preview", use_container_width=True)

    up_gt_r = st.file_uploader("Right Ground Truth Depth (.pfm)", type=["pfm"])
    if up_gt_r:
        gt_r_prev = read_pfm(up_gt_r.read());  up_gt_r.seek(0)
        st.image(vis_depth(gt_r_prev), caption="Right GT Depth Preview", use_container_width=True)

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
    # Step 3 — Crop ROI from Left Image
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Step 3: Crop Region of Interest")
    st.write("Define the bounding box of the object you want to recognise (pixels).")

    H, W = img_l.shape[:2]
    cr1, cr2, cr3, cr4 = st.columns(4)
    x0 = cr1.number_input("X start", 0, W - 2, 0, step=1)
    y0 = cr2.number_input("Y start", 0, H - 2, 0, step=1)
    x1 = cr3.number_input("X end",   int(x0) + 1, W, min(W, int(x0) + 100), step=1)
    y1 = cr4.number_input("Y end",   int(y0) + 1, H, min(H, int(y0) + 100), step=1)

    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    # Overlay rectangle on left image
    overlay = img_l.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
    crop_bgr = img_l[y0:y1, x0:x1].copy()

    ov1, ov2 = st.columns([3, 1])
    ov1.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
              caption="Left Image — ROI highlighted", use_container_width=True)
    ov2.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
              caption="Crop", use_container_width=True)

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

    aug_col1, aug_col2 = st.columns(2)
    aug_col1.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                   caption="Original Crop", use_container_width=True)
    aug_col2.image(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB),
                   caption="Augmented Crop", use_container_width=True)

    # -----------------------------------------------------------------------
    # Step 5 — Lock & Store
    # -----------------------------------------------------------------------
    st.divider()
    if st.button("🚀 Lock Data & Proceed to Benchmark"):
        st.session_state["pipeline_data"] = {
            "left":       img_l,
            "right":      img_r,
            "gt_left":    gt_depth_l,
            "gt_right":   gt_depth_r,
            "conf_raw":   conf_content,
            "crop":       crop_bgr,
            "crop_aug":   aug,
            "crop_bbox":  (x0, y0, x1, y1),
        }
        st.success("Data stored in session! Move to the 'Recognition' or 'Tuning' page.")

else:
    st.info("Please upload all 5 files (left image, right image, config, left GT, right GT) to proceed.")