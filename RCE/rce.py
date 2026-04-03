"""
Standard Restricted Coulomb Energy (RCE) classifier.

Prototype layer with spherical influence fields (Euclidean distance).
Training: prototype commitment + threshold modification.
Vectorized with numpy for practical speed on pixel-level data.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


class RCE:
    """
    RCE classifier following the standard formulation:
      - Each prototype: (center, radius, label).
      - Fires when d(x, center) < radius.
      - Training: commit new prototypes; shrink wrong-class radii.
    """

    def __init__(self, R_max: float = 100.0, default_label: str = "background"):
        self.R_max = float(R_max)
        self.default_label = default_label
        self.centers_ = None
        self.radii_ = None
        self.labels_ = None
        self.support_counts_ = None
        self.feature_dim_ = None

    @property
    def prototypes_(self):
        """Legacy access: list of (center, radius, label) tuples."""
        if self.centers_ is None:
            return []
        return list(zip(self.centers_, self.radii_, self.labels_))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n, f = X.shape
        self.feature_dim_ = int(f)

        unique_labels = np.unique(y)
        class_indices = {lbl: np.where(y == lbl)[0] for lbl in unique_labels}

        nearest_opp = np.full(n, self.R_max, dtype=np.float64)
        for lbl in unique_labels:
            own = class_indices[lbl]
            opp_idx = np.where(y != lbl)[0]
            if len(opp_idx) == 0:
                continue
            chunk = 5000
            for start in range(0, len(own), chunk):
                batch = own[start : start + chunk]
                D = cdist(X[batch], X[opp_idx], metric="euclidean")
                nearest_opp[batch] = np.minimum(nearest_opp[batch], D.min(axis=1))

        cap = min(n, 4096)
        P_centers = np.empty((cap, f), dtype=np.float64)
        P_radii = np.empty(cap, dtype=np.float64)
        P_labels = np.empty(cap, dtype=y.dtype)
        P_support = np.zeros(cap, dtype=np.int64)
        n_proto = 0

        for i in range(n):
            x = X[i]
            label = y[i]

            if n_proto > 0:
                dists = np.linalg.norm(P_centers[:n_proto] - x, axis=1)
                fired = dists < P_radii[:n_proto]
                L = P_labels[:n_proto]

                correct_firing = np.where((L == label) & fired)[0]
                if len(correct_firing) > 0:
                    best_idx = correct_firing[np.argmin(dists[correct_firing])]
                    P_support[best_idx] += 1
                    continue

                wrong_mask = (L != label) & fired
                if np.any(wrong_mask):
                    wrong_idx = np.where(wrong_mask)[0]
                    P_radii[wrong_idx] = dists[wrong_idx]

            r0 = min(float(nearest_opp[i]), self.R_max)
            r0 = max(r0, 1e-6)
            if n_proto >= cap:
                cap = cap * 2
                P_centers = np.resize(P_centers, (cap, f))
                P_radii = np.resize(P_radii, cap)
                P_labels = np.resize(P_labels, cap)
                P_support = np.resize(P_support, cap)
            P_centers[n_proto] = x
            P_radii[n_proto] = r0
            P_labels[n_proto] = label
            P_support[n_proto] = 1
            n_proto += 1

        self.centers_ = P_centers[:n_proto].copy()
        self.radii_ = P_radii[:n_proto].copy()
        self.labels_ = P_labels[:n_proto].copy()
        self.support_counts_ = P_support[:n_proto].copy()
        return self

    def _distances(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.centers_ is None or len(self.centers_) == 0:
            return X, np.empty((len(X), 0), dtype=np.float64)
        return X, cdist(X, self.centers_, metric="euclidean")

    def decision_details(self, X):
        """
        Return exact firing diagnostics for each sample.

        Margins are `radius - distance`, so positive values indicate firing.
        """
        X, D = self._distances(X)
        n = len(X)
        if self.centers_ is None or len(self.centers_) == 0:
            empty_idx = np.full(n, -1, dtype=int)
            empty_labels = np.full(n, self.default_label, dtype=object)
            return {
                "distances": D,
                "margins": D,
                "activated": np.zeros((n, 0), dtype=bool),
                "nearest_idx": empty_idx,
                "nearest_label": empty_labels,
                "nearest_firing_idx": empty_idx,
                "nearest_firing_label": empty_labels,
                "predicted_labels": empty_labels,
                "normalized_strength": np.zeros(n, dtype=np.float64),
            }

        margins = self.radii_[np.newaxis, :] - D
        activated = margins > 0
        nearest_idx = np.argmin(D, axis=1)
        nearest_label = self.labels_[nearest_idx]

        masked = np.where(activated, D, np.inf)
        nearest_firing_idx = np.argmin(masked, axis=1)
        nearest_firing_distance = masked[np.arange(n), nearest_firing_idx]
        has_fire = nearest_firing_distance < np.inf
        nearest_firing_idx = np.where(has_fire, nearest_firing_idx, -1)

        predicted_labels = np.full(n, self.default_label, dtype=object)
        predicted_labels[has_fire] = self.labels_[nearest_firing_idx[has_fire]]

        nearest_firing_label = np.full(n, self.default_label, dtype=object)
        nearest_firing_label[has_fire] = self.labels_[nearest_firing_idx[has_fire]]

        normalized_strength = np.zeros(n, dtype=np.float64)
        if np.any(has_fire):
            active_idx = nearest_firing_idx[has_fire]
            active_dist = D[np.where(has_fire)[0], active_idx]
            active_radius = self.radii_[active_idx]
            normalized_strength[has_fire] = np.clip(1.0 - (active_dist / active_radius), 0.0, 1.0)

        return {
            "distances": D,
            "margins": margins,
            "activated": activated,
            "nearest_idx": nearest_idx,
            "nearest_label": nearest_label,
            "nearest_firing_idx": nearest_firing_idx,
            "nearest_firing_label": nearest_firing_label,
            "predicted_labels": predicted_labels,
            "normalized_strength": normalized_strength,
        }

    def score_samples(self, X, positive_label, allow_nearest_margin: bool = False):
        """
        Heuristic confidence aligned with firing geometry.

        Scores use the normalized margin of the nearest firing prototype for the
        positive label. If no positive prototype fires and `allow_nearest_margin`
        is set, the score falls back to the clipped negative margin of the
        nearest positive prototype.
        """
        details = self.decision_details(X)
        scores = np.zeros(len(details["predicted_labels"]), dtype=np.float64)
        fire_mask = details["predicted_labels"] == positive_label
        scores[fire_mask] = details["normalized_strength"][fire_mask]

        if allow_nearest_margin and self.labels_ is not None and np.any(self.labels_ == positive_label):
            pos_mask = self.labels_ == positive_label
            pos_dist = details["distances"][:, pos_mask]
            pos_radius = self.radii_[pos_mask]
            pos_margin = pos_radius[np.newaxis, :] - pos_dist
            nearest_pos_margin = pos_margin.max(axis=1)
            fallback = np.clip(nearest_pos_margin / np.maximum(pos_radius.max(), 1e-6), 0.0, 1.0)
            scores = np.maximum(scores, fallback)
        return scores

    def predict(self, X):
        """Predict labels from firing prototypes."""
        details = self.decision_details(X)
        return details["predicted_labels"]

    def predict_proba(self, X, sigma: float = 0.1):
        """
        Heuristic probability mode: exp(-sigma * distance) summed by class.

        This is kept for compatibility, but the benchmark visuals prefer exact
        firing diagnostics from `decision_details` and `score_samples`.
        """
        X, D = self._distances(X)
        if self.centers_ is None or len(self.centers_) == 0:
            return np.ones((len(X), 1))

        labels = []
        for label in self.labels_:
            if label not in labels:
                labels.append(label)

        E = np.exp(-sigma * D)
        class_probs = np.zeros((len(X), len(labels)))
        for idx, label in enumerate(labels):
            mask = self.labels_ == label
            class_probs[:, idx] = E[:, mask].sum(axis=1)
        denom = class_probs.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return class_probs / denom
