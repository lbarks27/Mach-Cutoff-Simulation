"""Refractive-index field helpers.

This module keeps the original simple analytic field and also exposes the
new acoustic-grid field implementation from ``mach_cutoff``.
"""

from __future__ import annotations

import numpy as np

from mach_cutoff.atmosphere.acoustics import AcousticGridField


class RefractiveIndexField:
    """Simple analytic field used for quick tests.

    n(x) = n0 + linear_grad . x + sum_i A_i * exp(-||x-c_i||^2/(2*sigma_i^2))
    """

    def __init__(self, n0=1.0, gaussians=None, linear_grad=None):
        self.n0 = float(n0)
        self.gaussians = gaussians or []
        if linear_grad is None:
            self.linear_grad = np.zeros(3, dtype=float)
        else:
            self.linear_grad = np.asarray(linear_grad, dtype=float)

    def n(self, x):
        x = np.asarray(x, dtype=float)
        val = self.n0 + float(np.dot(self.linear_grad, x))
        for g in self.gaussians:
            A = float(g.get("amplitude", 0.0))
            c = np.asarray(g.get("center", np.zeros(3)), dtype=float)
            sigma = float(g.get("sigma", 1.0))
            r2 = np.sum((x - c) ** 2)
            val += A * np.exp(-r2 / (2.0 * sigma * sigma))
        return float(val)

    def grad_n(self, x):
        x = np.asarray(x, dtype=float)
        grad = np.array(self.linear_grad, dtype=float)
        for g in self.gaussians:
            A = float(g.get("amplitude", 0.0))
            c = np.asarray(g.get("center", np.zeros(3)), dtype=float)
            sigma = float(g.get("sigma", 1.0))
            diff = x - c
            exp_term = np.exp(-np.sum(diff * diff) / (2.0 * sigma * sigma))
            grad += A * exp_term * (-(diff) / (sigma * sigma))
        return grad


__all__ = ["RefractiveIndexField", "AcousticGridField"]
