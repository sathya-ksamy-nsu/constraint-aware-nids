"""Target models and a common train/eval interface.

Two target models spanning the differentiable / non-differentiable divide
(Section 4.2):

    * ``MLPClassifier``  - a PyTorch MLP, wrapped for ART so gradient attacks
      (FGSM/PGD) apply directly.
    * ``RandomForestModel`` - a scikit-learn random forest (non-differentiable),
      motivating black-box / decision-based attacks.

Both expose the same :class:`ModelInterface` API: ``fit``, ``predict``,
``predict_proba``, and ``to_art_classifier`` (the last returns an ART classifier
usable by the attack wrappers).

PyTorch and ART are imported lazily so the module can be *imported* in a minimal
environment; constructing/using the models still requires them installed.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


def set_global_seeds(seed: int) -> None:
    """Seed numpy, torch (if present) and python hash-based RNGs."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN where available (small perf cost, better repro).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


class ModelInterface(Protocol):
    """Common interface implemented by every target model."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ModelInterface": ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...
    def to_art_classifier(self): ...


# --------------------------------------------------------------------------- #
# MLP (PyTorch + ART)
# --------------------------------------------------------------------------- #
class MLPClassifier:
    """A small feedforward network for binary NIDS classification."""

    def __init__(
        self,
        n_features: int,
        hidden_dims=(128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 256,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        import torch

        self.n_features = n_features
        self.n_classes = 2
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        set_global_seeds(seed)
        self.net = self._build_net().to(self.device)
        self._art = None

    def _build_net(self):
        import torch.nn as nn

        # Cast inputs to the network's (float32) dtype. Some ART black-box
        # attacks (e.g. HopSkipJump) query the estimator with float64 arrays;
        # this leading cast keeps the Linear layers from hitting a
        # Double-vs-Float dtype mismatch.
        class _ToFloat32(nn.Module):
            def forward(self, x):
                return x.float()

        layers = [_ToFloat32()]
        prev = self.n_features
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers += [nn.Linear(prev, self.n_classes)]
        return nn.Sequential(*layers)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        set_global_seeds(self.seed)
        x_t = torch.tensor(np.asarray(x), dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y), dtype=torch.long)
        loader = DataLoader(
            TensorDataset(x_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        opt = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
        self._art = None  # invalidate cached wrapper
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self.net.eval()
        with torch.no_grad():
            x_t = torch.tensor(np.asarray(x), dtype=torch.float32, device=self.device)
            probs = F.softmax(self.net(x_t), dim=1).cpu().numpy()
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def to_art_classifier(self):
        """Wrap the network in an ART ``PyTorchClassifier``."""
        import torch
        from art.estimators.classification import PyTorchClassifier

        if self._art is not None:
            return self._art
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self._art = PyTorchClassifier(
            model=self.net,
            loss=loss_fn,
            optimizer=opt,
            input_shape=(self.n_features,),
            nb_classes=self.n_classes,
            clip_values=None,
            device_type="gpu" if self.device == "cuda" else "cpu",
        )
        return self._art


# --------------------------------------------------------------------------- #
# Random forest (scikit-learn + ART)
# --------------------------------------------------------------------------- #
class RandomForestModel:
    """A scikit-learn random forest wrapped for ART (black-box attacks)."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        n_jobs: int = -1,
        seed: int = 42,
    ):
        from sklearn.ensemble import RandomForestClassifier

        self.seed = seed
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=seed,
        )
        self._art = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        self.clf.fit(np.asarray(x), np.asarray(y))
        self._art = None
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(np.asarray(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict(np.asarray(x))

    def to_art_classifier(self):
        from art.estimators.classification import SklearnClassifier

        if self._art is None:
            self._art = SklearnClassifier(model=self.clf)
        return self._art


def build_model(name: str, n_features: int, cfg: dict):
    """Factory: build a target model by name using hyperparameters from cfg."""
    m = cfg.get("models", {})
    seed = int(cfg.get("seed", 42))
    if name == "mlp":
        p = m.get("mlp", {})
        return MLPClassifier(
            n_features=n_features,
            hidden_dims=p.get("hidden_dims", [128, 64]),
            dropout=p.get("dropout", 0.2),
            lr=p.get("lr", 1e-3),
            weight_decay=p.get("weight_decay", 1e-4),
            epochs=p.get("epochs", 20),
            batch_size=p.get("batch_size", 256),
            seed=seed,
        )
    if name in ("rf", "random_forest"):
        p = m.get("random_forest", {})
        return RandomForestModel(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", None),
            n_jobs=p.get("n_jobs", -1),
            seed=seed,
        )
    raise ValueError(f"Unknown model '{name}'. Choose 'mlp' or 'random_forest'.")
