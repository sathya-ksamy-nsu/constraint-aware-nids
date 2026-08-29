"""Adversarial attack wrappers (Section 4.3).

Wraps three attack families around a common :func:`run_attack` interface and
adds a *constrained* mode that projects each perturbation back onto the feasible
set defined by :mod:`src.constraints`:

    * ``fgsm``        - single-step gradient attack (ART FastGradientMethod).
    * ``pgd``         - iterative gradient attack (ART ProjectedGradientDescent).
    * ``hopskipjump`` - decision-based black-box attack (ART HopSkipJump); the
                        drop-in alternative ``zoo`` is also supported.
    * ``gan``         - a small self-contained GAN-based attack whose generator
                        learns additive perturbations pushing malicious samples
                        toward the benign class.

Constrained mode
----------------
ART attacks optimize in feature space. To respect realism we project after
generation (and, for the iterative attacks, this is a faithful approximation of
projecting inside the loop for the reporting purposes of this study). The
``ConstraintMask`` both projects the samples and reports the valid-sample rate.

ART / PyTorch are imported lazily inside functions, so importing this module in
a minimal environment does not fail.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import constraints as C


@dataclass
class AttackResult:
    """Everything downstream metrics need for one attack run."""

    x_clean: np.ndarray
    x_adv: np.ndarray
    y_true: np.ndarray
    validity_flags: Optional[np.ndarray] = None  # per-sample constraint validity


def _select_malicious(x: np.ndarray, y: np.ndarray):
    """Return the malicious subset (label==1) that attacks target for evasion."""
    y = np.asarray(y).astype(int)
    mask = y == C.MALICIOUS if hasattr(C, "MALICIOUS") else y == 1
    return x[mask], y[mask], mask


# --------------------------------------------------------------------------- #
# Gradient / black-box attacks via ART
# --------------------------------------------------------------------------- #
def _build_art_attack(name: str, art_classifier, cfg: dict):
    a = cfg.get("attacks", {})
    name = name.lower()
    if name == "fgsm":
        from art.attacks.evasion import FastGradientMethod

        p = a.get("fgsm", {})
        return FastGradientMethod(estimator=art_classifier, eps=p.get("eps", 0.1))
    if name == "pgd":
        from art.attacks.evasion import ProjectedGradientDescent

        p = a.get("pgd", {})
        return ProjectedGradientDescent(
            estimator=art_classifier,
            eps=p.get("eps", 0.1),
            eps_step=p.get("eps_step", 0.01),
            max_iter=p.get("max_iter", 40),
            verbose=False,
        )
    if name == "hopskipjump":
        from art.attacks.evasion import HopSkipJump

        p = a.get("hopskipjump", {})
        return HopSkipJump(
            classifier=art_classifier,
            targeted=False,
            max_iter=p.get("max_iter", 20),
            max_eval=p.get("max_eval", 1000),
            init_eval=p.get("init_eval", 100),
            verbose=False,
        )
    if name == "zoo":
        from art.attacks.evasion import ZooAttack

        p = a.get("zoo", {})
        return ZooAttack(
            classifier=art_classifier,
            max_iter=p.get("max_iter", 20),
            nb_parallel=p.get("nb_parallel", 1),
            verbose=False,
        )
    raise ValueError(f"Unknown ART attack '{name}'")


def run_art_attack(
    name: str,
    model,
    x: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    mask: Optional[C.ConstraintMask] = None,
) -> AttackResult:
    """Run an ART-backed attack (fgsm/pgd/hopskipjump/zoo) on malicious samples."""
    x_mal, y_mal, _ = _select_malicious(x, y)
    art_clf = model.to_art_classifier()
    attack = _build_art_attack(name, art_clf, cfg)
    x_adv = attack.generate(x=x_mal.astype(np.float32))
    x_adv = np.asarray(x_adv, dtype=float)

    return _finalize(x_mal, x_adv, y_mal, cfg, mask)


# --------------------------------------------------------------------------- #
# GAN-based attack (self-contained PyTorch generator)
# --------------------------------------------------------------------------- #
def run_gan_attack(
    model,
    x: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    mask: Optional[C.ConstraintMask] = None,
) -> AttackResult:
    """Train a small generator that perturbs malicious samples toward benign.

    The generator ``G(x, z)`` outputs a bounded additive perturbation
    ``delta = eps * tanh(...)``. It is trained to minimize the target model's
    predicted probability of the malicious class on ``x + delta`` (a distillation
    of the evasion objective). The target model is frozen and used as the
    (differentiable, for the MLP) discriminator/critic signal.

    For non-differentiable targets (random forest) the generator is trained
    against a differentiable surrogate MLP fit on the same data, then evaluated
    on the true target -- a standard transfer setup. Here we require the target
    to expose ``to_art_classifier``; if gradients are unavailable we fall back to
    a finite-difference-free surrogate only when the model is an MLP.
    """
    import torch
    import torch.nn as nn

    x_mal, y_mal, _ = _select_malicious(x, y)
    a = cfg.get("attacks", {}).get("gan", {})
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    n_features = x_mal.shape[1]
    latent_dim = a.get("latent_dim", 16)
    eps = float(a.get("eps", 0.1))
    hidden = a.get("generator_hidden", [64, 64])
    epochs = a.get("epochs", 30)
    batch_size = a.get("batch_size", 128)
    lr = a.get("lr", 2e-4)
    device = getattr(model, "device", "cpu")

    # Differentiable critic: reuse the MLP's network if present, else train a
    # quick surrogate MLP on (x, y) for transfer.
    critic = _get_differentiable_critic(model, x, y, cfg, device)

    # Generator: maps [x ; z] -> bounded perturbation.
    layers = []
    prev = n_features + latent_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, n_features), nn.Tanh()]
    gen = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)

    x_t = torch.tensor(x_mal, dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss()
    benign_target = torch.zeros(x_t.shape[0], dtype=torch.long, device=device)

    gen.train()
    for _ in range(epochs):
        perm = torch.randperm(x_t.shape[0], device=device)
        for i in range(0, x_t.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            xb = x_t[idx]
            z = torch.randn(xb.shape[0], latent_dim, device=device)
            delta = eps * gen(torch.cat([xb, z], dim=1))
            logits = critic(xb + delta)
            # Push toward benign class (0) while keeping perturbation small.
            loss = ce(logits, benign_target[: xb.shape[0]]) + 0.01 * delta.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    gen.eval()
    with torch.no_grad():
        z = torch.randn(x_t.shape[0], latent_dim, device=device)
        delta = eps * gen(torch.cat([x_t, z], dim=1))
        x_adv = (x_t + delta).cpu().numpy().astype(float)

    return _finalize(x_mal, x_adv, y_mal, cfg, mask)


def _get_differentiable_critic(model, x, y, cfg, device):
    """Return a differentiable torch module scoring malicious-vs-benign.

    If ``model`` is an MLP we reuse its frozen network. Otherwise we train a
    quick surrogate MLP (transfer attack) so the GAN has gradients to follow.
    """
    import torch

    net = getattr(model, "net", None)
    if net is not None:
        for p in net.parameters():
            p.requires_grad_(False)
        return net

    # Surrogate for non-differentiable targets (e.g. random forest).
    from .models import MLPClassifier

    surrogate = MLPClassifier(
        n_features=x.shape[1], epochs=cfg.get("models", {}).get("mlp", {}).get("epochs", 20),
        seed=int(cfg.get("seed", 42)),
        device=device,
    )
    surrogate.fit(x, y)
    for p in surrogate.net.parameters():
        p.requires_grad_(False)
    return surrogate.net


# --------------------------------------------------------------------------- #
# Shared finalization: constraint projection + validity flags.
# --------------------------------------------------------------------------- #
def _finalize(
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    y_true: np.ndarray,
    cfg: dict,
    mask: Optional[C.ConstraintMask],
) -> AttackResult:
    constrained = bool(cfg.get("constraints", {}).get("enabled", False)) and mask is not None
    validity = None
    if mask is not None:
        mode = cfg.get("constraints", {}).get("mode", "project")
        if constrained and mode == "project":
            x_adv = mask.project(x_adv, x_clean)
        # Always record validity of whatever we ended up with.
        validity = mask.is_valid(x_adv, x_clean)
    return AttackResult(
        x_clean=x_clean, x_adv=x_adv, y_true=y_true, validity_flags=validity
    )


def run_attack(
    name: str,
    model,
    x: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    mask: Optional[C.ConstraintMask] = None,
) -> AttackResult:
    """Dispatch to the appropriate attack implementation by name."""
    name = name.lower()
    if name == "gan":
        return run_gan_attack(model, x, y, cfg, mask)
    return run_art_attack(name, model, x, y, cfg, mask)
