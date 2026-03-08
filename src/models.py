"""
src/models.py  —  Central Model Registry
=========================================
Downloads backbone weights **once** from the internet (PyTorch Hub / timm),
freezes every feature-extraction layer, and caches the result in RAM with
Streamlit's ``@st.cache_resource``.

Strategy
--------
1. **Freeze the Backbone**   → ``requires_grad = False`` on every parameter.
   The backbone is a pure feature extractor — no gradient updates, ever.
2. **Cache the Resource**    → ``@st.cache_resource`` keeps the heavy model
   in RAM even when you switch pages.
3. **Define the Head**       → ``RecognitionHead``: a tiny sklearn
   LogisticRegression that takes the backbone's feature vector and
   produces a recognition score.  Lives only in ``st.session_state``.
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import timm
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Device selection (MPS > CUDA > CPU)
# ---------------------------------------------------------------------------
DEVICE = (
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)

# ---------------------------------------------------------------------------
# Shared ImageNet preprocessing
# ---------------------------------------------------------------------------
_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ===================================================================
#  Base class
# ===================================================================
class _FrozenBackbone:
    """Shared helpers: freeze, normalise activation maps."""

    DIM: int = 0                     # overridden by subclasses

    # --- freeze every parameter ---
    def _freeze(self, model: nn.Module) -> nn.Module:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model.to(DEVICE)

    # --- public interface ---
    def get_features(self, img_bgr: np.ndarray) -> np.ndarray:
        """Return a 1-D float32 feature vector for *img_bgr* (BGR uint8)."""
        raise NotImplementedError

    def get_activation_maps(self, img_bgr: np.ndarray,
                            n_maps: int = 6) -> list[np.ndarray]:
        """Return *n_maps* normalised float32 spatial activation maps."""
        raise NotImplementedError

    @staticmethod
    def _norm(m: np.ndarray) -> np.ndarray:
        lo, hi = m.min(), m.max()
        return ((m - lo) / (hi - lo + 1e-5)).astype(np.float32)


# ===================================================================
#  ResNet-18
# ===================================================================
class ResNet18Backbone(_FrozenBackbone):
    """ResNet-18 downloaded from PyTorch Hub, frozen, classifier removed."""

    DIM = 512

    def __init__(self):
        full = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = self._freeze(full)
        self.extractor = nn.Sequential(*list(full.children())[:-1]).to(DEVICE)
        self.transform = _IMAGENET_TRANSFORM

    def get_features(self, img_bgr):
        t = self.transform(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            return self.extractor(t.unsqueeze(0).to(DEVICE)).cpu().numpy().flatten()

    def get_activation_maps(self, img_bgr, n_maps=6):
        cap = {}
        hook = self.backbone.layer4.register_forward_hook(
            lambda _m, _i, o: cap.update(feat=o))
        t = self.transform(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            self.backbone(t.unsqueeze(0).to(DEVICE))
        hook.remove()
        acts = cap["feat"][0].cpu().numpy()
        return [self._norm(acts[i]) for i in range(min(n_maps, acts.shape[0]))]


# ===================================================================
#  MobileNetV3-Small
# ===================================================================
class MobileNetV3Backbone(_FrozenBackbone):
    """MobileNetV3-Small from PyTorch Hub, frozen, classifier = Identity."""

    DIM = 576

    def __init__(self):
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        self._freeze(self.backbone)
        self.transform = _IMAGENET_TRANSFORM

    def get_features(self, img_bgr):
        t = self.transform(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            return self.backbone(t.unsqueeze(0).to(DEVICE)).cpu().numpy().flatten()

    def get_activation_maps(self, img_bgr, n_maps=6):
        cap = {}
        hook = self.backbone.features[-1].register_forward_hook(
            lambda _m, _i, o: cap.update(feat=o))
        t = self.transform(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            self.backbone(t.unsqueeze(0).to(DEVICE))
        hook.remove()
        acts = cap["feat"][0].cpu().numpy()
        return [self._norm(acts[i]) for i in range(min(n_maps, acts.shape[0]))]


# ===================================================================
#  MobileViT-XXS
# ===================================================================
class MobileViTBackbone(_FrozenBackbone):
    """MobileViT-XXS from timm (Apple Research), frozen."""

    DIM = 320

    def __init__(self):
        self.backbone = timm.create_model(
            "mobilevit_xxs.cvnets_in1k", pretrained=True, num_classes=0)
        self._freeze(self.backbone)
        cfg = timm.data.resolve_model_data_config(self.backbone)
        self.transform = timm.data.create_transform(**cfg, is_training=False)

    def _to_tensor(self, img_bgr):
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return self.transform(pil).unsqueeze(0).to(DEVICE)

    def get_features(self, img_bgr):
        with torch.no_grad():
            return self.backbone(self._to_tensor(img_bgr)).cpu().numpy().flatten()

    def get_activation_maps(self, img_bgr, n_maps=6):
        cap = {}
        hook = self.backbone.stages[-1].register_forward_hook(
            lambda _m, _i, o: cap.update(feat=o))
        with torch.no_grad():
            self.backbone(self._to_tensor(img_bgr))
        hook.remove()
        acts = cap["feat"][0].cpu().numpy()
        return [self._norm(acts[i]) for i in range(min(n_maps, acts.shape[0]))]


# ===================================================================
#  Lightweight Head  (lives in session state, never on disk)
# ===================================================================
class RecognitionHead:
    """
    A tiny trainable layer on top of a frozen backbone.
    Wraps sklearn ``LogisticRegression`` for binary classification.
    Stored in ``st.session_state`` — never saved to disk.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=C, max_iter=max_iter)
        self.is_trained = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        return self

    def predict(self, features: np.ndarray):
        """Return *(label, confidence)* for a single feature vector."""
        probs = self.model.predict_proba([features])[0]
        idx = int(np.argmax(probs))
        return self.model.classes_[idx], probs[idx]

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.model.classes_


# ===================================================================
#  Cached loaders  —  @st.cache_resource keeps models in RAM
# ===================================================================
@st.cache_resource
def get_resnet() -> ResNet18Backbone:
    """Download & freeze ResNet-18.  Stays in RAM across page switches."""
    return ResNet18Backbone()

@st.cache_resource
def get_mobilenet() -> MobileNetV3Backbone:
    """Download & freeze MobileNetV3-Small.  Stays in RAM."""
    return MobileNetV3Backbone()

@st.cache_resource
def get_mobilevit() -> MobileViTBackbone:
    """Download & freeze MobileViT-XXS.  Stays in RAM."""
    return MobileViTBackbone()


# ===================================================================
#  BACKBONES  —  The Registry Dict
# ===================================================================
BACKBONES = {
    "ResNet-18": {
        "loader":     get_resnet,
        "dim":        ResNet18Backbone.DIM,
        "hook_layer": "layer4 (last conv block)",
    },
    "MobileNetV3": {
        "loader":     get_mobilenet,
        "dim":        MobileNetV3Backbone.DIM,
        "hook_layer": "features[-1] (last features block)",
    },
    "MobileViT-XXS": {
        "loader":     get_mobilevit,
        "dim":        MobileViTBackbone.DIM,
        "hook_layer": "stages[-1] (last transformer stage)",
    },
}
