# Claude Code Implementation Prompt
# Recognition-BenchMark — Full Restructure

---

## Context

This is a Streamlit-based stereo-vision benchmarking platform called **Recognition-BenchMark**. It compares a custom hand-crafted feature extractor called **RCE (Relative Contextual Encoding)** against CNN-based deep learning approaches for object recognition and depth estimation.

### Current Project Structure

```
app.py                          ← Landing page (home)
pages/
├── 2_Data_Lab.py               ← Stage 1
├── 3_Feature_Lab.py            ← Stage 2
├── 4_Model_Tuning.py           ← Stage 3
├── 5_Localization_Lab.py       ← Stage 4
├── 6_RealTime_Detection.py     ← Stage 5
├── 7_Evaluation.py             ← Stage 6
└── 8_Stereo_Geometry.py        ← Stage 7
src/
├── config.py                   ← App configuration constants
├── detectors/
│   ├── base.py                 ← Base detector class
│   ├── rce/
│   │   ├── __init__.py
│   │   └── features.py         ← RCE feature extractor (DO NOT MODIFY)
│   ├── mobilenet.py            ← MobileNetV3 detector (DO NOT MODIFY)
│   ├── mobilevit.py            ← MobileViT detector (DO NOT MODIFY)
│   ├── resnet.py               ← ResNet-18 detector (DO NOT MODIFY)
│   ├── orb.py                  ← ORB detector (DO NOT MODIFY)
│   └── yolo.py                 ← YOLOv8 detector (DO NOT MODIFY)
├── localization.py             ← Localization strategies (DO NOT MODIFY)
└── models.py                   ← Model loading utilities (DO NOT MODIFY)
models/
├── mobilenet_v3_head.pkl
├── mobilenet_v3.pth
├── mobilevit_head.pkl
├── mobilevit_xxs.pth
├── orb_reference.pkl
├── resnet18_head.pkl
├── resnet18.pth
└── yolov8n.pt
data/
└── middlebury/                 ← Bundled dataset (already present)
```

**The entire `src/` directory must not be modified.** All detector logic, feature extraction, localization strategies, and model loading are already implemented there. The pages in `pages/` import from `src/` and must be migrated to the new `tabs/` structure while continuing to import from `src/`.

---

## Critical Bug To Fix

**Data leakage through circular evaluation.** Currently the detection/recognition stage runs on the same left image used to define the training ROI. This is scientifically invalid — the model is tested on its own training source.

**The fix:**
- In the Stereo pipeline: train on LEFT image crop → detect on RIGHT image
- In the Generalisation pipeline: train on image 1 crop → detect on image 2

This must be propagated through session state so every stage after Data Lab knows which image is for training (source) and which is for testing (target).

---

## Target Architecture

### File Structure After Refactor

```
app.py                              ← REPLACE: routing controller + home page
tabs/
├── stereo/
│   ├── __init__.py
│   ├── data_lab.py                 ← NEW: replaces pages/2_Data_Lab.py for stereo
│   ├── feature_lab.py              ← MIGRATE: from pages/3_Feature_Lab.py
│   ├── model_tuning.py             ← MIGRATE: from pages/4_Model_Tuning.py
│   ├── localization.py             ← MIGRATE: from pages/5_Localization_Lab.py
│   ├── detection.py                ← MIGRATE + FIX: from pages/6_RealTime_Detection.py
│   ├── evaluation.py               ← MIGRATE: from pages/7_Evaluation.py
│   └── stereo_depth.py             ← MIGRATE: from pages/8_Stereo_Geometry.py
├── generalisation/
│   ├── __init__.py
│   ├── data_lab.py                 ← NEW: generalisation-specific data loading
│   ├── feature_lab.py              ← ADAPT: stereo version with gen_pipeline keys
│   ├── model_tuning.py             ← ADAPT: stereo version with gen_pipeline keys
│   ├── localization.py             ← ADAPT: stereo version with gen_pipeline keys
│   ├── detection.py                ← ADAPT + FIX: stereo version with gen_pipeline keys
│   └── evaluation.py               ← ADAPT: stereo version with gen_pipeline keys
utils/
└── middlebury_loader.py            ← NEW: dataset scanning, loading, parsing
src/                                ← DO NOT TOUCH: all detector/model logic stays here
pages/                              ← DELETE after migration is complete and verified
data/
└── middlebury/                     ← Already present, do not modify
```

---

## Part 1 — app.py Routing Controller

Replace the existing `app.py` entirely. The new `app.py` is a **routing controller** that:

1. Sets page config (keep existing title/icon/layout)
2. Builds the sidebar navigation manually using session state
3. Renders the correct module based on navigation state
4. Preserves the existing landing page content (pipeline overview, models, depth info)

### Sidebar Logic

```python
import streamlit as st

# Top-level navigation
st.sidebar.title("🦅 Recognition BenchMark")

top_section = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📷 Stereo + Depth", "🌍 Generalisation"],
    key="top_nav"
)

if top_section == "🏠 Home":
    # render home/landing page content inline in app.py
    render_home()

elif top_section == "📷 Stereo + Depth":
    stereo_stage = st.sidebar.radio(
        "Pipeline Stage",
        [
            "🧪 1 · Data Lab",
            "🔬 2 · Feature Lab",
            "⚙️ 3 · Model Tuning",
            "🔍 4 · Localization",
            "🎯 5 · Detection",
            "📈 6 · Evaluation",
            "📐 7 · Stereo Depth"
        ],
        key="stereo_stage"
    )
    # import and call the appropriate render() function from tabs/stereo/

elif top_section == "🌍 Generalisation":
    gen_stage = st.sidebar.radio(
        "Pipeline Stage",
        [
            "🧪 1 · Data Lab",
            "🔬 2 · Feature Lab",
            "⚙️ 3 · Model Tuning",
            "🔍 4 · Localization",
            "🎯 5 · Detection",
            "📈 6 · Evaluation"
        ],
        key="gen_stage"
    )
    # import and call the appropriate render() function from tabs/generalisation/
```

### Stage Guard Pattern

Every stage except Data Lab must check if the previous stage is complete. Use this pattern at the top of each stage's `render()` function:

```python
def render():
    pipe = st.session_state.get("stereo_pipeline", {})
    if "train_image" not in pipe:
        st.warning("⚠️ Complete **Data Lab** first before accessing this stage.")
        st.stop()
    # ... rest of stage logic
```

### Session State Namespacing

**Critical:** The two pipelines must never share session state keys.

- Stereo pipeline uses: `st.session_state["stereo_pipeline"]` — a dict containing all stereo stage data
- Generalisation pipeline uses: `st.session_state["gen_pipeline"]` — a dict containing all generalisation stage data

Within each dict, use consistent keys:
```python
# Stereo pipeline dict keys
stereo_pipeline = {
    "train_image":    np.ndarray,   # LEFT image — used for ROI + training
    "test_image":     np.ndarray,   # RIGHT image — used for detection
    "calib":          dict,         # parsed calibration parameters
    "disparity_gt":   np.ndarray,   # ground truth disparity (optional, may be None)
    "roi":            dict,         # {"x", "y", "w", "h", "label"}
    "crop":           np.ndarray,   # cropped ROI from train_image
    "crop_aug":       list,         # augmented crop variants
    "active_modules": list,         # RCE modules ["intensity", "sobel", "spectral"]
    "rce_head":       object,       # trained LogisticRegression
    "cnn_heads":      dict,         # {"ResNet-18": ..., "MobileNetV3": ..., "MobileViT-XXS": ...}
    "rce_dets":       list,         # detection results on test_image
    "cnn_dets":       dict,         # detection results per CNN model
    "source":         str,          # "middlebury" or "custom"
    "scene_name":     str,          # Middlebury scene name (if source == "middlebury")
}

# Generalisation pipeline dict keys — same structure minus calib/disparity_gt
gen_pipeline = {
    "train_image":    np.ndarray,   # im0.png from training scene variant
    "test_image":     np.ndarray,   # im0.png from test scene variant
    "roi":            dict,
    "crop":           np.ndarray,
    "crop_aug":       list,
    "active_modules": list,
    "rce_head":       object,
    "cnn_heads":      dict,
    "rce_dets":       list,
    "cnn_dets":       dict,
    "source":         str,          # "middlebury" or "custom"
    "scene_group":    str,          # e.g. "artroom" (Middlebury only)
    "train_scene":    str,          # e.g. "artroom1" (Middlebury only)
    "test_scene":     str,          # e.g. "artroom2" (Middlebury only)
}
```

---

## Part 2 — Middlebury Loader Utility

Create `utils/middlebury_loader.py` with the following functions:

### `scan_dataset_root(root_path: str) -> list[str]`
- Scan root directory for valid scene folders
- A valid scene must contain: `im0.png`, `im1.png`, `calib.txt`
- Return sorted list of scene names

### `get_scene_groups(root_path: str) -> dict`
- Scan all valid scenes and group them by scene base name (strip trailing digit)
- e.g. `artroom1`, `artroom2` → group `"artroom"`
- Return dict: `{"artroom": ["artroom1", "artroom2"], "curule": ["curule1", "curule2", "curule3"], ...}`
- Used by Tab 2 to present scene group selection then variant selection

### `get_available_views(scene_path: str) -> list[dict]`
- This dataset has NO multi-exposure variants (no im0E.png etc.)
- Function kept for future compatibility but always returns single entry:
  `[{"suffix": "", "label": "Primary (im0/im1)"}]`

### `load_stereo_pair(scene_path: str, view_suffix: str = '') -> dict`
- Load `im0{suffix}.png` as left image (train_image)
- Load `im1{suffix}.png` as right image (test_image)
- Load and parse `calib.txt`
- Load `disp0.pfm` if it exists (else None)
- Return dict with keys: `left`, `right`, `calib`, `disparity_gt`

### `load_single_view(scene_path: str, view_suffix: str) -> np.ndarray`
- Load and return a single image: `im0{suffix}.png`
- Used by generalisation tab when selecting individual views

### `parse_calib(calib_path: str) -> dict`
Parse Middlebury `calib.txt` format:
```
cam0=[fx 0 cx; 0 fy cy; 0 0 1]
cam1=[fx 0 cx; 0 fy cy; 0 0 1]
doffs=x_offset
baseline=Bmm
width=W
height=H
ndisp=N
vmin=v
vmax=v
```
Extract and return: `{"fx": float, "baseline": float, "doffs": float, "width": int, "height": int, "ndisp": int}`

Use regex to extract `fx` from the camera matrix string: first numeric value after `cam0=[`.

### `load_pfm(filepath: str) -> np.ndarray`
Load PFM (Portable FloatMap) file:
- Read header line (`PF` = color, `Pf` = grayscale)
- Read dimensions line
- Read scale factor (negative = little-endian)
- Read float32 binary data
- Flip vertically (PFM origin is bottom-left)
- Return numpy array

### Dataset Root Resolution

The dataset is **bundled directly in the repo** at `./data/middlebury/`. No user configuration needed.

```python
DEFAULT_MIDDLEBURY_ROOT = "./data/middlebury"
```

If the path does not exist or contains no valid scenes, show a clear error. This should not happen in normal deployment since the data is bundled.

### Bundled Scenes Reference

The following 10 scene folders are bundled, forming 4 scene groups:

```python
BUNDLED_SCENES = {
    "artroom":   ["artroom1",  "artroom2"],
    "curule":    ["curule1",   "curule2",   "curule3"],
    "skates":    ["skates1",   "skates2"],
    "skiboots":  ["skiboots1", "skiboots2", "skiboots3"],
}
```

Each folder contains exactly: `im0.png`, `im1.png`, `disp0.pfm`, `disp1.pfm`, `calib.txt`.

There are **no multi-exposure variants** (no `im0E.png` etc.) — the scene groups ARE the multi-condition variants. `artroom1` and `artroom2` are different captures of the same artroom scene.

---

## Part 3 — Tab 1: Stereo Data Lab

Create `tabs/stereo/data_lab.py` with a `render()` function.

### Data Source Selection

```python
st.header("🧪 Data Lab — Stereo + Depth")
st.info("**How this works:** Define your object of interest in the LEFT image. The system trains on it and attempts to recognise it in the RIGHT image — a genuinely different viewpoint.")

source = st.radio(
    "Data source",
    ["📦 Middlebury Dataset", "📁 Upload your own files"],
    horizontal=True
)
```

### If Middlebury Selected

```
1. Scan dataset root → show selectbox of available scenes
2. Auto-load im0.png (train/left) and im1.png (test/right)
3. Auto-load calib.txt → parse parameters
4. Auto-load disp0.pfm if present
5. Display LEFT image (train) and RIGHT image (test) side by side
6. Show parsed calibration parameters in an expander
7. Show ground truth disparity colormap if available
```

Show a clear visual label:
- Left image labeled: `🟦 TRAIN IMAGE (Left)` 
- Right image labeled: `🟥 TEST IMAGE (Right)`

### If Custom Upload Selected

```
- Left image uploader (png/jpg) → labeled as TRAIN IMAGE
- Right image uploader (png/jpg) → labeled as TEST IMAGE  
- Calibration file uploader (txt) — REQUIRED for depth estimation
- PFM ground truth uploader (pfm) — optional, disables depth evaluation if missing
- If calibration file not provided: show warning "Depth estimation will be disabled"
```

### ROI Definition

After images are loaded (either source):

```
1. Display LEFT (train) image only for ROI definition
2. Use streamlit-cropper or manual coordinate inputs for ROI selection
   - If streamlit-cropper available: use it
   - Fallback: four number_input widgets for x, y, w, h
3. Text input for class label (default: "object")
4. Show cropped ROI preview
5. "Lock Data Lab" button → saves everything to st.session_state["stereo_pipeline"]
```

### Data Augmentation

After ROI is locked, show augmentation controls (preserve existing augmentation logic):
- Rotation, brightness, contrast, noise, blur, flip
- Preview augmented crops
- "Apply Augmentation" button

### What Gets Saved to Session State

```python
st.session_state["stereo_pipeline"] = {
    "train_image": left_image,      # numpy array, BGR
    "test_image":  right_image,     # numpy array, BGR  ← KEY FIX
    "calib":       calib_dict,      # parsed params or None
    "disparity_gt": disp_gt,        # numpy array or None
    "roi":         {"x":x, "y":y, "w":w, "h":h, "label":label},
    "crop":        cropped_roi,
    "crop_aug":    augmented_list,
    "source":      "middlebury" or "custom",
    "scene_name":  scene_name or "",
}
```

---

## Part 4 — Tab 2: Generalisation Data Lab

Create `tabs/generalisation/data_lab.py` with a `render()` function.

### Key Difference From Stereo Data Lab

- No calibration file
- No depth estimation
- Two images can be completely independent OR different views from Middlebury
- Goal is testing appearance generalisation, not stereo geometry

### Data Source Selection

```python
st.header("🧪 Data Lab — Generalisation")
st.info("**How this works:** Train on one image, test on a completely different image of the same object. No stereo geometry — pure recognition generalisation.")

source = st.radio(
    "Data source",
    ["📦 Middlebury Multi-View", "📁 Upload your own files"],
    horizontal=True
)
```

### If Middlebury Selected

```
1. Show scene group selector: ["artroom", "curule", "skates", "skiboots"]
2. Based on selected group, show available variants:
   - artroom → [artroom1, artroom2]
   - curule  → [curule1, curule2, curule3]
   - skates  → [skates1, skates2]
   - skiboots → [skiboots1, skiboots2, skiboots3]
3. Two selectboxes:
   - "Training scene" → user picks one variant (e.g. artroom1)
   - "Test scene"     → user picks a DIFFERENT variant (e.g. artroom2)
   - Validate: training scene ≠ test scene, show error if same selected
4. Load train_scene/im0.png as train_image
5. Load test_scene/im0.png as test_image
   (NOTE: both are LEFT images im0.png, from different scene variants)
6. Display both side by side with clear labels
```

Show labels:
- Train image: `🟦 TRAIN IMAGE (artroom1)`
- Test image: `🟥 TEST IMAGE (artroom2)`

Also show an explanation: *"Both images show the same scene type captured under different conditions. The model trains on one variant and must recognise the same object class in the other — testing genuine appearance generalisation."*

### If Custom Upload Selected

```
- Train image uploader → labeled TRAIN IMAGE
- Test image uploader → labeled TEST IMAGE
- No calibration, no PFM needed
- Simple, low barrier
```

### ROI Definition and Augmentation

Same as stereo data lab but on the TRAIN image only. Save to `st.session_state["gen_pipeline"]` with same key structure (minus calib and disparity_gt).

---

## Part 5 — Migrate Existing Pipeline Stages

The existing pages (Feature Lab, Model Tuning, Localization, Detection, Evaluation, Stereo Depth) must be migrated into the new `tabs/` structure.

### Migration Rules

1. **Read each existing page file** before migrating it:
   - `pages/2_Data_Lab.py` → split into `tabs/stereo/data_lab.py` and `tabs/generalisation/data_lab.py`
   - `pages/3_Feature_Lab.py` → `tabs/stereo/feature_lab.py` (adapt for `tabs/generalisation/feature_lab.py`)
   - `pages/4_Model_Tuning.py` → `tabs/stereo/model_tuning.py`
   - `pages/5_Localization_Lab.py` → `tabs/stereo/localization.py`
   - `pages/6_RealTime_Detection.py` → `tabs/stereo/detection.py` ← apply data leakage fix here
   - `pages/7_Evaluation.py` → `tabs/stereo/evaluation.py`
   - `pages/8_Stereo_Geometry.py` → `tabs/stereo/stereo_depth.py`

2. **Each page becomes a module** with a `render()` function — wrap all existing page code inside `def render(): ...`

3. **Update all session state reads** — the existing pages use `st.session_state.get("pipeline_data", {})` or similar flat keys. Replace with namespaced dicts:
   - Stereo stages: `st.session_state.get("stereo_pipeline", {})`
   - Generalisation stages: `st.session_state.get("gen_pipeline", {})`

4. **Preserve all imports from `src/`** — every `from src.xxx import yyy` in the existing pages must be kept exactly as-is

5. **Detection stage fix** — the critical data leakage fix:
   - OLD: detection runs on the same image used for training ROI definition
   - NEW: `image_to_scan = pipe["test_image"]` ← the other image

### Stage Guard Template

```python
def render():
    pipe = st.session_state.get("stereo_pipeline", {})  # or gen_pipeline
    
    required_keys = ["train_image", "test_image", "crop_aug"]
    missing = [k for k in required_keys if k not in pipe]
    if missing:
        st.warning("⚠️ Complete the **Data Lab** stage first.")
        st.info("Go to: 📷 Stereo + Depth → 🧪 Data Lab")
        st.stop()
```

### Specific Migration Notes Per Stage

**Feature Lab (Stage 2):**
- Visualise features extracted from `pipe["crop"]` (from train_image)
- No changes needed beyond session state key updates

**Model Tuning (Stage 3):**
- Training data comes from `pipe["crop_aug"]` (augmented crops of train_image ROI)
- Negatives sampled from `pipe["train_image"]` (not test_image)
- No changes needed beyond session state key updates

**Detection (Stage 5) — CRITICAL FIX:**
- Run sliding window on `pipe["test_image"]` NOT `pipe["train_image"]`
- Add a visual reminder in the UI: *"Running detection on TEST image (right/second image)"*
- For stereo: show test_image (right) with detection results
- For generalisation: show test_image (different exposure) with detection results

**Stereo Depth (Stage 7 — stereo only):**
- Requires `pipe["calib"]` to be not None
- If calib is None (custom upload without calibration): show warning and disable depth computation
- If disparity_gt is None: skip ground truth comparison, show note
- Otherwise: preserve existing StereoSGBM logic entirely

---

## Part 6 — Session Status Widget Update

Update the session status display in `app.py` (home page) to show status for BOTH pipelines:

```python
st.header("📋 Session Status")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Stereo Pipeline")
    stereo = st.session_state.get("stereo_pipeline", {})
    stereo_checks = {
        "Data loaded":        "train_image" in stereo and "test_image" in stereo,
        "ROI defined":        "roi" in stereo,
        "Augmentation done":  "crop_aug" in stereo,
        "Modules locked":     "active_modules" in stereo,
        "Models trained":     "rce_head" in stereo,
        "Detection run":      "rce_dets" in stereo,
    }
    for label, done in stereo_checks.items():
        st.markdown(f"{'✅' if done else '⬜'} {label}")

with col2:
    st.subheader("🌍 Generalisation Pipeline")
    gen = st.session_state.get("gen_pipeline", {})
    gen_checks = {
        "Data loaded":        "train_image" in gen and "test_image" in gen,
        "ROI defined":        "roi" in gen,
        "Augmentation done":  "crop_aug" in gen,
        "Modules locked":     "active_modules" in gen,
        "Models trained":     "rce_head" in gen,
        "Detection run":      "rce_dets" in gen,
    }
    for label, done in gen_checks.items():
        st.markdown(f"{'✅' if done else '⬜'} {label}")
```

---

## Part 7 — Shared Utility Modules

All core logic lives in `src/` and must be imported identically by both `tabs/stereo/` and `tabs/generalisation/` stages. Do not duplicate or move anything from `src/`.

Key imports used by the stage files:
```python
from src.detectors.rce.features import RCEExtractor        # RCE feature extraction
from src.detectors.resnet import ResNetDetector             # ResNet-18
from src.detectors.mobilenet import MobileNetDetector       # MobileNetV3
from src.detectors.mobilevit import MobileViTDetector       # MobileViT-XXS
from src.detectors.orb import ORBDetector                   # ORB keypoint matching
from src.detectors.yolo import YOLODetector                 # YOLOv8
from src.localization import LocalizationStrategy           # All 5 localization strategies
from src.models import load_model                           # Model loading from models/
from src.config import *                                    # App constants
```

The `models/` directory contains pre-trained weights referenced by `src/models.py`. Do not move or rename any files in `models/`.

---

## Part 8 — Landing Page (Home)

The existing landing page content in `app.py` must be preserved and rendered when `top_nav == "🏠 Home"`. Extract it into a `render_home()` function within `app.py`.

Update the Pipeline Overview section to reflect the new two-pipeline structure:

```
🗺️ Pipeline Overview

This platform provides two evaluation pipelines:

📷 Stereo + Depth (7 stages)
Train on the LEFT image, detect in the RIGHT image, estimate metric depth.
Evaluates RCE in a constrained stereo-vision scenario.

🌍 Generalisation (6 stages)  
Train on one view/exposure, detect in a different view/exposure.
Evaluates RCE's robustness to appearance variation.

Both pipelines compare: RCE · ResNet-18 · MobileNetV3-Small · MobileViT-XXS · ORB
```

Update the bottom caption: *"Navigate using the sidebar → Choose a pipeline to begin"*

---

## Implementation Order

Implement in this exact order to avoid breaking dependencies:

1. `utils/middlebury_loader.py` — no dependencies, can be tested in isolation
2. `app.py` — routing shell, import stubs for tabs not yet created
3. `tabs/stereo/data_lab.py` — foundation of stereo pipeline
4. `tabs/generalisation/data_lab.py` — foundation of generalisation pipeline
5. Migrate existing stages into `tabs/stereo/` — feature_lab, model_tuning, localization, detection (with fix), evaluation, stereo_depth
6. Create `tabs/generalisation/` stages — reuse stereo logic with gen_pipeline session keys
7. Update home page session status widget

---

## What NOT To Change

- **Entire `src/` directory** — all detector logic, RCE, CNN backbones, ORB, localization, model loading
- **Entire `models/` directory** — pre-trained weights
- **Entire `data/` directory** — Middlebury dataset
- **`notebooks/`, `training/`, `scripts/`** — development artifacts, leave untouched
- **`Dockerfile`, `packages.txt`, `requirements.txt`** — deployment config (add `streamlit-cropper` to requirements.txt only)
- The visual style and markdown descriptions of each stage
- The models description tabs on the home page (RCE, ResNet-18, MobileNetV3, MobileViT-XXS)
- The depth estimation explanation and LaTeX formula

---

## Dependencies To Add

Add to `requirements.txt` if not already present:
```
streamlit-cropper   # for ROI selection (optional but preferred)
```

No other new dependencies are needed. The Middlebury loader uses only `numpy`, `opencv-python`, `re`, `os`, and `pathlib` — all already present.

---

## Testing Checklist

After implementation, verify:

- [ ] Home page renders with both pipeline status widgets
- [ ] Clicking "Stereo + Depth" shows 7-stage sub-navigation
- [ ] Clicking "Generalisation" shows 6-stage sub-navigation
- [ ] Clicking stages before Data Lab shows guard warning
- [ ] Middlebury loader finds scenes from `./data/middlebury/`
- [ ] Stereo Data Lab correctly assigns LEFT → train_image, RIGHT → test_image
- [ ] Generalisation Data Lab correctly assigns View1 → train_image, View2 → test_image
- [ ] Detection stage uses `pipe["test_image"]` in BOTH pipelines
- [ ] Stereo and Generalisation pipelines do not share session state
- [ ] Depth estimation gracefully disabled when calib is None
- [ ] All existing stage logic works after migration
