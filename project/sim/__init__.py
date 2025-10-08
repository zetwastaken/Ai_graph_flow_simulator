"""Simulation utilities package.

This package contains the core data generator as well as placeholder
modules for anomaly definitions.  Importing from this package makes
available the high‑level :func:`project.sim.generator.simulate` function.
"""

from .generator import simulate  # re‑export for convenience

__all__ = ["simulate"]