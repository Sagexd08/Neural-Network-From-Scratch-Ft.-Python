"""
initialization.py
=================

How a layer's weights are set before training begins.

Why initialization matters at all
---------------------------------
It looks like a detail — the optimizer will move the weights anyway — but the
starting point decides whether learning happens at all.  Two failure modes:

**All zeros.**  If every weight in a layer is 0, every unit in that layer
computes the identical output, receives the identical gradient, and takes the
identical step.  They stay identical forever.  A 128-unit layer has the power
of a 1-unit layer.  This is the *symmetry problem*, and it is why biases (one
per unit, already distinguished by the weights feeding them) may safely start
at zero but weights may not.

**Wrong scale.**  Consider the variance of a unit's pre-activation
``z = sum over fan_in inputs of w_i x_i``.  If the ``w_i`` are independent with
variance ``Var(w)`` and the inputs have variance ``Var(x)``, then::

    Var(z) = fan_in * Var(w) * Var(x)

So the signal is multiplied by ``fan_in * Var(w)`` at every layer.  If that
factor is below 1, activations shrink geometrically with depth until they are
numerically zero (*vanishing*); above 1, they explode.  The same factor
applies to gradients on the way back.

Every scheme below is one answer to: *choose ``Var(w)`` so the factor is 1.*

    Xavier/Glorot: Var(w) = 2 / (fan_in + fan_out)
        Solves it for both directions at once (forward needs 1/fan_in,
        backward needs 1/fan_out; take the harmonic compromise).  Derived
        assuming a symmetric activation with derivative ~1 near the origin,
        so it is the right default for tanh and sigmoid.

    He/Kaiming: Var(w) = 2 / fan_in
        ReLU zeroes out half its inputs, which halves the variance passing
        through.  The factor of 2 compensates exactly.  Use with ReLU and
        its relatives.

The rule of thumb: **He for ReLU-family, Xavier for tanh/sigmoid.**
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict

from .tensor import Tensor
from .utils import get_rng

__all__ = [
    "zeros_init",
    "ones_init",
    "constant_init",
    "random_uniform_init",
    "random_normal_init",
    "xavier_uniform_init",
    "xavier_normal_init",
    "he_uniform_init",
    "he_normal_init",
    "lecun_normal_init",
    "get_initializer",
    "INITIALIZERS",
]


def _new(fan_in: int, fan_out: int, sample: Callable[[], float]) -> Tensor:
    """Build a ``(fan_in, fan_out)`` weight matrix by calling ``sample()``.

    The shape convention is ``W[i][j] = weight from input i to output j``,
    which is what makes the forward pass a plain ``X @ W`` with X holding one
    sample per row.
    """
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError(
            f"layer dimensions must be positive, got fan_in={fan_in}, fan_out={fan_out}"
        )
    return Tensor([sample() for _ in range(fan_in * fan_out)], (fan_in, fan_out))


def zeros_init(fan_in: int, fan_out: int, rng: random.Random | None = None) -> Tensor:
    """All zeros.

    Correct for **biases**, wrong for **weights** — see the symmetry problem
    in the module docstring.  Provided so the failure can be demonstrated.
    """
    return _new(fan_in, fan_out, lambda: 0.0)


def ones_init(fan_in: int, fan_out: int, rng: random.Random | None = None) -> Tensor:
    """All ones.  Used by BatchNorm's scale parameter, not by Dense weights."""
    return _new(fan_in, fan_out, lambda: 1.0)


def constant_init(value: float) -> Callable[..., Tensor]:
    """Return an initializer that fills with ``value``."""
    def init(fan_in: int, fan_out: int, rng: random.Random | None = None) -> Tensor:
        return _new(fan_in, fan_out, lambda: float(value))
    return init


def random_uniform_init(fan_in: int, fan_out: int, rng: random.Random | None = None,
                        limit: float = 0.05) -> Tensor:
    """Uniform on ``[-limit, limit]`` with a fixed, fan-independent limit.

    The naive approach.  It breaks symmetry, but because the scale ignores
    ``fan_in``, signal variance drifts with depth — fine for a 2-layer toy,
    poor for anything deeper.
    """
    r = rng or get_rng()
    return _new(fan_in, fan_out, lambda: r.uniform(-limit, limit))


def random_normal_init(fan_in: int, fan_out: int, rng: random.Random | None = None,
                       std: float = 0.05) -> Tensor:
    """Gaussian with a fixed standard deviation.  Same caveat as above."""
    r = rng or get_rng()
    return _new(fan_in, fan_out, lambda: r.gauss(0.0, std))


def xavier_uniform_init(fan_in: int, fan_out: int, rng: random.Random | None = None) -> Tensor:
    """Glorot uniform.

    Target ``Var(w) = 2/(fan_in + fan_out)``.  A uniform distribution on
    ``[-L, L]`` has variance ``L^2/3``, so setting ``L^2/3 = 2/(fan_in+fan_out)``
    gives::

        L = sqrt(6 / (fan_in + fan_out))

    That 6 is not magic — it is the 3 from the uniform variance times the 2
    from the target.
    """
    r = rng or get_rng()
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return _new(fan_in, fan_out, lambda: r.uniform(-limit, limit))


def xavier_normal_init(fan_in: int, fan_out: int, rng: random.Random | None = None) -> Tensor:
    """Glorot normal: Gaussian with ``std = sqrt(2/(fan_in + fan_out))``."""
    r = rng or get_rng()
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return _new(fan_in, fan_out, lambda: r.gauss(0.0, std))


def he_uniform_init(fan_in: int, fan_out: int, rng: random.Random | None = None) -> Tensor:
    """He uniform.

    Target ``Var(w) = 2/fan_in``; with the same ``L^2/3`` uniform variance::

        L = sqrt(6 / fan_in)
    """
    r = rng or get_rng()
    limit = math.sqrt(6.0 / fan_in)
    return _new(fan_in, fan_out, lambda: r.uniform(-limit, limit))


def he_normal_init(fan_in: int, fan_out: int, rng: random.Random | None = None) -> Tensor:
    """He normal: Gaussian with ``std = sqrt(2/fan_in)``.

    The default for ReLU networks, and the default for :class:`Dense` here.
    """
    r = rng or get_rng()
    std = math.sqrt(2.0 / fan_in)
    return _new(fan_in, fan_out, lambda: r.gauss(0.0, std))


def lecun_normal_init(fan_in: int, fan_out: int, rng: random.Random | None = None) -> Tensor:
    """LeCun normal: ``std = sqrt(1/fan_in)``.

    Preserves forward variance only (no ReLU correction, no backward
    compromise).  This is the plain ``sqrt(1/fan_in)`` formulation.
    """
    r = rng or get_rng()
    std = math.sqrt(1.0 / fan_in)
    return _new(fan_in, fan_out, lambda: r.gauss(0.0, std))


INITIALIZERS: Dict[str, Callable[..., Tensor]] = {
    "zeros": zeros_init,
    "zero": zeros_init,
    "ones": ones_init,
    "random": random_uniform_init,
    "random_uniform": random_uniform_init,
    "random_normal": random_normal_init,
    "xavier": xavier_uniform_init,
    "glorot": xavier_uniform_init,
    "xavier_uniform": xavier_uniform_init,
    "glorot_uniform": xavier_uniform_init,
    "xavier_normal": xavier_normal_init,
    "glorot_normal": xavier_normal_init,
    "he": he_normal_init,
    "kaiming": he_normal_init,
    "he_normal": he_normal_init,
    "he_uniform": he_uniform_init,
    "lecun": lecun_normal_init,
    "lecun_normal": lecun_normal_init,
}


def get_initializer(spec) -> Callable[..., Tensor]:
    """Resolve ``"he"`` (or a callable) to an initializer function."""
    if callable(spec):
        return spec
    if isinstance(spec, str):
        key = spec.lower().strip()
        if key not in INITIALIZERS:
            raise ValueError(
                f"unknown initialization {spec!r}; available: "
                f"{', '.join(sorted(set(INITIALIZERS)))}"
            )
        return INITIALIZERS[key]
    raise TypeError(f"cannot interpret {spec!r} as an initializer")
