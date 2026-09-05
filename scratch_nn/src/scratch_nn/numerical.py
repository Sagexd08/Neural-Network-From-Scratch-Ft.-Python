"""
numerical.py
============

Gradient checking: independent verification that the hand-derived calculus in
:mod:`scratch_nn.layers` is actually correct.

This module is the quality gate for the entire project.  Backpropagation is
easy to get *subtly* wrong — a transpose in the wrong place, a missing factor
of 2, a sum over the wrong axis — and a subtly wrong gradient still trains,
just worse.  It produces no exception and no obvious symptom.  Gradient
checking catches all of it.

The idea
--------
The derivative is defined as a limit of difference quotients.  The naive
forward difference::

    f'(x) ~= [f(x + h) - f(x)] / h

has error O(h): its Taylor expansion is ``f'(x) + (h/2)f''(x) + ...``, and
that ``h/2`` term does not vanish.  The *central* difference is far better::

    f'(x) ~= [f(x + h) - f(x - h)] / (2h)

Expanding both terms::

    f(x+h) = f(x) + h f'(x) + (h^2/2) f''(x) + (h^3/6) f'''(x) + ...
    f(x-h) = f(x) - h f'(x) + (h^2/2) f''(x) - (h^3/6) f'''(x) + ...

Subtracting cancels every *even* term, including the ``h^2/2 f''`` one::

    f(x+h) - f(x-h) = 2h f'(x) + (h^3/3) f'''(x) + ...

so dividing by ``2h`` leaves an error of O(h^2).  With ``h = 1e-5`` that is
an approximation error around 1e-10 — small enough that any disagreement
above ~1e-7 means a real bug, not float noise.

Choosing h.  There are two competing errors.  Truncation error falls as h
shrinks (O(h^2) above).  *Round-off* error grows as h shrinks, because
``f(x+h) - f(x-h)`` subtracts two nearly-equal float64 values and loses
significant digits (catastrophic cancellation).  ``h = 1e-5`` sits near the
sweet spot; ``h = 1e-10`` would be dominated by round-off and report failures
everywhere.

The comparison metric
---------------------
Absolute difference is useless across scales (1e-3 is huge for a gradient of
1e-6 and negligible for one of 1e3).  We use a relative error::

                    |analytical - numerical|
    rel_error = ----------------------------------
                 max(1, |analytical|, |numerical|)

The ``1`` in the denominator is deliberate: the pure relative form
``|a-n|/(|a|+|n|)`` blows up to 1.0 when both gradients are legitimately
near zero, producing false alarms.  Flooring the denominator at 1 makes the
metric fall back to an absolute comparison in that regime, which is the
right behaviour.

Interpreting the number::

    < 1e-7   gradients agree — the implementation is correct
    < 1e-5   acceptable, especially with ReLU (see the kink caveat below)
    > 1e-4   almost certainly a bug

The ReLU caveat.  At a kink, ``f`` is not differentiable, and if ``x`` is
within ``h`` of 0 the two probes land on opposite sides of it, so the
numerical estimate averages two different slopes and legitimately disagrees.
This is a limitation of the *check*, not a bug in the gradient.  The
functions here report such cases rather than failing on them.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .tensor import Tensor
from .utils import format_float, format_table

__all__ = [
    "relative_error",
    "numerical_gradient",
    "GradientCheckResult",
    "check_gradients",
    "check_layer_gradients",
    "gradient_check_report",
]


def relative_error(analytical: float, numerical: float) -> float:
    """The scale-aware disagreement metric described in the module docstring."""
    denom = max(1.0, abs(analytical), abs(numerical))
    return abs(analytical - numerical) / denom


def numerical_gradient(loss_fn: Callable[[], float], parameter: Tensor, index: int,
                       h: float = 1e-5) -> float:
    """Estimate ``dL/d(parameter[index])`` by central difference.

    ``loss_fn`` must recompute the loss from scratch each call, reading the
    current parameter values — that is what makes the perturbation visible.

    The original value is restored in a ``finally`` block so a raising
    ``loss_fn`` cannot leave the model corrupted.
    """
    data = parameter.data
    original = data[index]
    try:
        data[index] = original + h
        loss_plus = loss_fn()

        data[index] = original - h
        loss_minus = loss_fn()
    finally:
        data[index] = original

    return (loss_plus - loss_minus) / (2.0 * h)


class GradientCheckResult:
    """Outcome of a gradient check, with enough detail to debug a failure."""

    def __init__(self, max_error: float, mean_error: float, num_checked: int,
                 failures: List[Dict[str, Any]], samples: List[Dict[str, Any]],
                 tolerance: float, skipped_kinks: int = 0):
        self.max_error = max_error
        self.mean_error = mean_error
        self.num_checked = num_checked
        self.failures = failures
        self.samples = samples
        self.tolerance = tolerance
        self.skipped_kinks = skipped_kinks

    @property
    def passed(self) -> bool:
        """True when every checked gradient was within tolerance."""
        return not self.failures

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Gradient check: {status}",
            f"  parameters checked : {self.num_checked}",
            f"  max relative error : {self.max_error:.3e}  (tolerance {self.tolerance:.0e})",
            f"  mean relative error: {self.mean_error:.3e}",
        ]
        if self.skipped_kinks:
            lines.append(
                f"  skipped            : {self.skipped_kinks} points near a "
                f"non-differentiable kink"
            )
        if self.failures:
            lines.append(f"  failures           : {len(self.failures)}")
            for f in self.failures[:5]:
                lines.append(
                    f"    {f['param']}[{f['index']}]: analytical="
                    f"{f['analytical']:+.6e}  numerical={f['numerical']:+.6e}  "
                    f"rel_err={f['error']:.3e}"
                )
            if len(self.failures) > 5:
                lines.append(f"    ... and {len(self.failures) - 5} more")
        return "\n".join(lines)

    def table(self, limit: int = 12) -> str:
        """A side-by-side listing of analytical vs numerical gradients."""
        rows = []
        for s in self.samples[:limit]:
            rows.append([
                f"{s['param']}[{s['index']}]",
                format_float(s["analytical"], 12, 6),
                format_float(s["numerical"], 12, 6),
                f"{s['error']:.2e}",
                "ok" if s["error"] <= self.tolerance else "FAIL",
            ])
        return format_table(
            rows, headers=["parameter", "analytical", "numerical", "rel err", ""]
        )

    def __repr__(self) -> str:
        return (f"GradientCheckResult(passed={self.passed}, "
                f"max_error={self.max_error:.2e}, checked={self.num_checked})")


def check_gradients(model, X, y, h: float = 1e-5, tolerance: float = 1e-6,
                    max_checks_per_tensor: int = 12, seed: Optional[int] = None,
                    skip_kinks: bool = True, verbose: bool = False) -> GradientCheckResult:
    """Verify a model's analytical gradients against finite differences.

    Parameters
    ----------
    model
        A compiled :class:`~scratch_nn.model.Sequential`.
    X, y
        A small batch.  Small matters: every probe costs two full forward
        passes, so 4-8 samples is plenty.
    h
        Central-difference step.  1e-5 balances truncation against round-off.
    tolerance
        Maximum acceptable relative error.
    max_checks_per_tensor
        Probe this many random entries per parameter tensor rather than all
        of them; checking a 128x128 weight matrix exhaustively would need
        32768 forward passes.
    skip_kinks
        Skip probes where the loss is non-smooth between the two evaluation
        points (a ReLU boundary), which the central difference cannot handle
        by construction.

    Notes
    -----
    Dropout is disabled during the check.  It re-randomises its mask each
    forward pass, so the "same" loss function would differ between the ``+h``
    and ``-h`` probes and the estimate would be pure noise.

    Returns
    -------
    GradientCheckResult
    """
    import random as _random
    from .layers import BatchNorm, Dropout

    rng = _random.Random(seed if seed is not None else 12345)

    # 1. Turn dropout off — it would make loss_fn non-deterministic.
    dropout_rates = []
    for layer in model.layers:
        if isinstance(layer, Dropout):
            dropout_rates.append((layer, layer.rate))
            layer.rate = 0.0
            layer.scale = 1.0

    # BatchNorm behaves differently in the two modes, and its backward pass
    # needs the batch statistics that only a training-mode forward caches.  So
    # a model containing it must be checked in training mode throughout — the
    # function stays deterministic because dropout is already disabled and the
    # batch is fixed.
    mode = any(isinstance(l, BatchNorm) for l in model.layers)

    # Running statistics are updated on every training-mode forward pass, which
    # would make the loss drift between probes.  Freeze them by restoring the
    # buffers after each call.
    bn_layers = [l for l in model.layers if isinstance(l, BatchNorm)]
    bn_saved = [(l, list(l.running_mean.data), list(l.running_var.data))
                for l in bn_layers]

    def _restore_bn() -> None:
        for layer, mean, var in bn_saved:
            layer.running_mean.data[:] = mean
            layer.running_var.data[:] = var

    try:
        # 2. Analytical gradients: one forward + one backward pass.
        y_pred = model.forward(X, training=mode)
        _restore_bn()
        model.zero_grad()
        model.backward(y_pred, y)
        analytical = [g.copy() for g in model.gradients()]
        parameters = model.parameters()

        # 3. The loss closure the numerical probes will call.  It must recompute
        #    from the current parameter values, which is what makes the
        #    perturbation visible.
        def loss_fn() -> float:
            pred = model.forward(X, training=mode)
            _restore_bn()
            return model.compute_loss(pred, y)

        errors: List[float] = []
        failures: List[Dict[str, Any]] = []
        samples: List[Dict[str, Any]] = []
        skipped = 0

        for p_idx, (param, grad) in enumerate(zip(parameters, analytical)):
            name = f"param{p_idx}{param.shape}"
            n = param.size
            if n <= max_checks_per_tensor:
                indices = list(range(n))
            else:
                indices = rng.sample(range(n), max_checks_per_tensor)

            for idx in indices:
                num = numerical_gradient(loss_fn, param, idx, h)
                ana = grad.data[idx]

                if skip_kinks and _near_kink(loss_fn, param, idx, h):
                    skipped += 1
                    continue

                err = relative_error(ana, num)
                errors.append(err)
                record = {"param": name, "index": idx, "analytical": ana,
                          "numerical": num, "error": err}
                samples.append(record)
                if err > tolerance:
                    failures.append(record)
                if verbose:
                    flag = "FAIL" if err > tolerance else "ok"
                    print(f"  {name}[{idx}]: ana={ana:+.6e} num={num:+.6e} "
                          f"err={err:.2e} {flag}")

        max_err = max(errors) if errors else 0.0
        mean_err = sum(errors) / len(errors) if errors else 0.0
        return GradientCheckResult(max_err, mean_err, len(errors), failures,
                                   samples, tolerance, skipped)
    finally:
        # 4. Always restore dropout, even if a probe raised.
        for layer, rate in dropout_rates:
            layer.rate = rate
            layer.scale = 1.0 / (1.0 - rate) if rate > 0 else 1.0


def _near_kink(loss_fn: Callable[[], float], parameter: Tensor, index: int,
               h: float) -> bool:
    """Detect a non-differentiable point between the two probe locations.

    Test for local linearity: for a smooth function, the midpoint value
    should equal the average of the endpoints to within O(h^2).  A kink (ReLU
    switching on, or an L1 penalty changing sign) breaks that badly.

    The threshold is generous — this only ever suppresses a probe, so a false
    positive costs coverage, never correctness.
    """
    data = parameter.data
    original = data[index]
    try:
        data[index] = original + h
        f_plus = loss_fn()
        data[index] = original - h
        f_minus = loss_fn()
        data[index] = original
        f_mid = loss_fn()
    finally:
        data[index] = original

    # For a smooth f this quantity is h^2 * |f''|, so with h=1e-5 it is about
    # 1e-10 times the local curvature.  At a kink one probe lands on a
    # different linear piece and the quantity jumps to O(h) — five orders of
    # magnitude larger.  Comparing against `h` (scaled by the loss size to stay
    # dimensionless) separates the two cases cleanly.
    curvature = abs(f_plus + f_minus - 2.0 * f_mid)
    scale = max(abs(f_mid), 1.0)
    return curvature > h * scale * 1e-3


def check_layer_gradients(layer, x: Tensor, upstream_grad: Optional[Tensor] = None,
                          h: float = 1e-5, tolerance: float = 1e-6,
                          max_checks: int = 20) -> GradientCheckResult:
    """Gradient-check a single layer in isolation, with no model around it.

    A surrogate scalar "loss" is used::

        L = sum over all elements of  layer.forward(x) * upstream_grad

    Its derivative with respect to the layer output is exactly
    ``upstream_grad``, so passing that into ``layer.backward`` gives gradients
    that must match finite differences of ``L``.  This isolates one layer's
    calculus from the rest of the stack, which makes failures far easier to
    localise.
    """
    import random as _random
    from .layers import BatchNorm, Dropout

    rng = _random.Random(999)

    # BatchNorm's backward pass needs the batch statistics that only a
    # training-mode forward caches, so it must be checked in training mode.
    # Dropout would make the loss non-deterministic between probes, so it is
    # always checked in inference mode.
    mode = isinstance(layer, BatchNorm)

    # Freeze the running statistics: a training-mode forward updates them, and
    # the drift would corrupt the finite-difference estimate.
    bn_saved = None
    if isinstance(layer, BatchNorm):
        bn_saved = (list(layer.running_mean.data), list(layer.running_var.data))

    def _restore_bn() -> None:
        if bn_saved is not None:
            layer.running_mean.data[:] = bn_saved[0]
            layer.running_var.data[:] = bn_saved[1]

    out = layer.forward(x, training=mode)
    _restore_bn()
    if upstream_grad is None:
        upstream_grad = Tensor([rng.uniform(-1.0, 1.0) for _ in range(out.size)],
                               out.shape)

    def loss_fn() -> float:
        o = layer.forward(x, training=mode)
        _restore_bn()
        return math.fsum(o.data[i] * upstream_grad.data[i] for i in range(o.size))

    layer.zero_grad()
    layer.forward(x, training=mode)
    _restore_bn()
    layer.backward(upstream_grad)
    analytical = [g.copy() for g in layer.gradients()]
    parameters = layer.parameters()

    errors: List[float] = []
    failures: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []

    for p_idx, (param, grad) in enumerate(zip(parameters, analytical)):
        name = f"param{p_idx}{param.shape}"
        indices = (list(range(param.size)) if param.size <= max_checks
                   else rng.sample(range(param.size), max_checks))
        for idx in indices:
            num = numerical_gradient(loss_fn, param, idx, h)
            ana = grad.data[idx]
            err = relative_error(ana, num)
            errors.append(err)
            rec = {"param": name, "index": idx, "analytical": ana,
                   "numerical": num, "error": err}
            samples.append(rec)
            if err > tolerance:
                failures.append(rec)

    max_err = max(errors) if errors else 0.0
    mean_err = sum(errors) / len(errors) if errors else 0.0
    return GradientCheckResult(max_err, mean_err, len(errors), failures, samples,
                               tolerance)


def gradient_check_report(model, X, y, **kwargs) -> str:
    """Run :func:`check_gradients` and format a human-readable report."""
    result = check_gradients(model, X, y, **kwargs)
    lines = [
        "=" * 62,
        "NUMERICAL GRADIENT CHECK",
        "=" * 62,
        "",
        "Comparing analytical backpropagation against central differences:",
        "",
        "    numerical  =  [L(w + h) - L(w - h)] / (2h)",
        "    rel_error  =  |analytical - numerical| / max(1, |a|, |n|)",
        "",
        result.table(),
        "",
        result.summary(),
        "=" * 62,
    ]
    return "\n".join(lines)
