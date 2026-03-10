import cv2
import numpy as np
from src.detectors.rce.features import REGISTRY


def build_rce_vector(img_bgr, active_modules):
    """Build an RCE feature vector from a BGR image patch.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR image patch.
    active_modules : dict
        Mapping ``{module_key: bool}`` indicating which RCE modules are on.

    Returns
    -------
    np.ndarray
        1-D float32 feature vector (10 bins per active module).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    vec = []
    for key, meta in REGISTRY.items():
        if active_modules.get(key, False):
            v, _ = meta["fn"](gray)
            vec.extend(v)
    return np.array(vec, dtype=np.float32)
