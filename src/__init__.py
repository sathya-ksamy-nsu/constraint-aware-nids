"""Source package for the constraint-aware adversarial-NIDS evaluation harness.

Submodules are intentionally NOT imported here. ``constraints`` and ``metrics``
depend only on NumPy and can be imported/tested without PyTorch or ART, while
``models``, ``attacks``, ``defense``, ``data_loader`` and ``evaluate`` pull in
the heavy ML stack. Import submodules explicitly, e.g.::

    from src import constraints, metrics          # lightweight, always works
    from src import models, attacks               # requires torch + ART
"""

__all__ = [
    "constraints",
    "metrics",
    "data_loader",
    "models",
    "attacks",
    "defense",
    "evaluate",
]

__version__ = "0.1.0"
