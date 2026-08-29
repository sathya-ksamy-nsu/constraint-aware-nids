"""Adversarial-training defense (Section 4.5).

Implements the standard Madry et al. (2018) defense: retrain the target MLP on a
mixture of clean and PGD-generated adversarial examples, then hand the hardened
model back so the full attack matrix can be re-run against it.

Only the differentiable MLP target is adversarially trained here (the random
forest is defended via transfer in practice; that extension is left as a TODO).
PyTorch and ART are imported lazily.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import constraints as C
from .models import MLPClassifier, set_global_seeds


def adversarial_train_mlp(
    model: MLPClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    mask: Optional[C.ConstraintMask] = None,
) -> MLPClassifier:
    """Adversarially train an MLP in place and return it.

    For each epoch we generate PGD adversarial examples for a fraction of the
    training data (``defense.adversarial_training.ratio``) and train on the
    clean+adversarial mixture. When a constraint ``mask`` is supplied and
    constraints are enabled, adversarial examples are projected onto the feasible
    set before training, so the model is hardened against *realistic* attacks.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    d = cfg.get("defense", {}).get("adversarial_training", {})
    ratio = float(d.get("ratio", 0.5))
    epochs = int(d.get("epochs", cfg.get("models", {}).get("mlp", {}).get("epochs", 20)))
    seed = int(cfg.get("seed", 42))
    set_global_seeds(seed)

    from art.attacks.evasion import ProjectedGradientDescent

    p = cfg.get("attacks", {}).get("pgd", {})

    device = model.device
    x_t = torch.tensor(np.asarray(x_train), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y_train), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=model.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    opt = torch.optim.Adam(
        model.net.parameters(), lr=model.lr, weight_decay=model.weight_decay
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    constrained = bool(cfg.get("constraints", {}).get("enabled", False)) and mask is not None

    for _ in range(epochs):
        model.net.train()
        # Rebuild the ART attack each epoch so it tracks current weights.
        art_clf = model.to_art_classifier()
        attack = ProjectedGradientDescent(
            estimator=art_clf,
            eps=p.get("eps", 0.1),
            eps_step=p.get("eps_step", 0.01),
            max_iter=p.get("max_iter", 40),
            verbose=False,
        )
        for xb, yb in loader:
            xb_np = xb.numpy()
            k = max(1, int(ratio * xb_np.shape[0]))
            idx = np.random.choice(xb_np.shape[0], size=k, replace=False)
            adv = attack.generate(x=xb_np[idx].astype(np.float32))
            if constrained:
                adv = mask.project(np.asarray(adv, dtype=float), xb_np[idx])
            mixed = xb_np.copy()
            mixed[idx] = adv
            xb_mixed = torch.tensor(mixed, dtype=torch.float32, device=device)
            yb_dev = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model.net(xb_mixed), yb_dev)
            loss.backward()
            opt.step()

    model._art = None  # invalidate cached wrapper after weight updates
    return model
