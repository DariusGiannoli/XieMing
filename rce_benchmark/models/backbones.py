"""Backbone feature extractors for the benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass
class BackboneSpec:
    model_id: str
    backbone_name: str
    expected_dim: int
    weight_path: Path | None
    loader_name: str

BACKBONE_SPECS = {
    "resnet18": BackboneSpec(
        model_id="resnet18",
        backbone_name="resnet18",
        expected_dim=512,
        weight_path=MODEL_DIR / "resnet18.pth",
        loader_name="torchvision_resnet18",
    ),
    "mobilenetv3": BackboneSpec(
        model_id="mobilenetv3",
        backbone_name="mobilenetv3",
        expected_dim=576,
        weight_path=MODEL_DIR / "mobilenet_v3.pth",
        loader_name="torchvision_mobilenet_v3_small",
    ),
    "mobilevit_xxs": BackboneSpec(
        model_id="mobilevit_xxs",
        backbone_name="mobilevit_xxs",
        expected_dim=320,
        weight_path=MODEL_DIR / "mobilevit_xxs.pth",
        loader_name="timm_mobilevit_xxs",
    ),
}


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class FrozenBackboneExtractor:
    """Frozen feature extractor that loads local weights when available."""

    def __init__(self, backbone_id: str):
        self.spec = BACKBONE_SPECS[backbone_id]
        self.device = _device()
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

    def _load_model(self):
        import timm
        import torch
        import torch.nn as nn
        import torchvision.models as tv_models

        if self.spec.model_id == "resnet18":
            model = tv_models.resnet18(weights=None)
            state_path = self.spec.weight_path
            if state_path and state_path.exists():
                state_dict = torch.load(state_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
            else:
                raise RuntimeError(
                    f"Missing local weights for {self.spec.model_id}: {state_path}. "
                    "Download backbone weights before running backbone benchmarks."
                )
            return nn.Sequential(*list(model.children())[:-1])

        if self.spec.model_id == "mobilenetv3":
            model = tv_models.mobilenet_v3_small(weights=None)
            state_path = self.spec.weight_path
            if state_path and state_path.exists():
                state_dict = torch.load(state_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
            else:
                raise RuntimeError(
                    f"Missing local weights for {self.spec.model_id}: {state_path}. "
                    "Download backbone weights before running backbone benchmarks."
                )
            model.classifier = nn.Identity()
            return model

        if self.spec.model_id == "mobilevit_xxs":
            state_path = self.spec.weight_path
            if not state_path or not state_path.exists():
                raise RuntimeError(
                    f"Missing local weights for {self.spec.model_id}: {state_path}. "
                    "Download backbone weights before running backbone benchmarks."
                )
            model = timm.create_model("mobilevit_xxs.cvnets_in1k", pretrained=False, num_classes=0)
            state_dict = torch.load(state_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            return model

        raise KeyError(f"Unsupported backbone: {self.spec.model_id}")

    def extract(self, patch_bgr: np.ndarray) -> np.ndarray:
        """Extract a feature vector from a patch."""
        import cv2
        import numpy as np
        import torch
        import torchvision.transforms as transforms

        imagenet_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        if self.spec.model_id == "mobilevit_xxs":
            from PIL import Image

            rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            tensor = imagenet_transform(np.array(pil_image)).unsqueeze(0).to(self.device)
        else:
            rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            tensor = imagenet_transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(tensor)
        return feature.cpu().numpy().astype(np.float32).flatten()
