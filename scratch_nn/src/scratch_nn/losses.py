"""
losses.py
=========

Loss functions: the single number that training tries to make small, and the
gradient that says which way to move.

Every loss provides:

``forward(y_pred, y_true) -> float``
    The scalar loss, averaged over the batch.

``backward(y_pred, y_true) -> Tensor``
    ``dL/dy_pred``, the same shape as ``y_pred``.

The batch-size convention
-------------------------
Both methods average over the batch.  ``forward`` divides the summed loss by
``N``; ``backward`` divides the gradient by the same ``N``.  Keeping the two
consistent is what makes the gradient the true derivative of the reported
loss — and it is exactly what a finite-difference gradient check verifies.

Averaging (rather than summing) means the gradient magnitude does not depend
on batch size, so you can change ``batch_size`` without retuning the learning
rate.
"""

from __future__ import annotations

import math
from typing import Dict, Type

from .tensor import Tensor, TensorShapeError
from .utils import EPS, safe_log

__all__ = [
    "Loss",
    "MeanSquaredError",
    "MeanAbsoluteError",
    "BinaryCrossEntropy",
    "CategoricalCrossEntropy",
    "Huber",
    "get_loss",
    "LOSSES",
]


def _check_shapes(y_pred: Tensor, y_true: Tensor, name: str) -> None:
    """Both arguments must be the same shape, and non-empty."""
    if y_pred.shape != y_true.shape:
        raise TensorShapeError(
            f"{name}: predictions have shape {y_pred.shape} but targets have "
            f"{y_true.shape}. For one output per sample, targets should be "
            f"(batch, 1) — a plain list of labels needs reshaping."
        )
    if y_pred.size == 0:
        raise TensorShapeError(f"{name}: cannot compute a loss over an empty batch")


class Loss:
    """Base class for losses."""

    name: str = "loss"
    #: True when this loss expects a softmax layer immediately before it, and
    #: supports the fused ``dZ = p - y`` shortcut.
    pairs_with_softmax: bool = False

    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        raise NotImplementedError

    def backward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> float:
        return self.forward(y_pred, y_true)

    def config(self) -> Dict[str, object]:
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class MeanSquaredError(Loss):
    """MSE — the default for regression.

    Forward::

        L = (1/N) * sum over all elements of (y_pred - y_true)^2

    Backward.  Differentiating the sum with respect to one prediction leaves
    only that prediction's term::

        dL/dy_pred_i = (2/N) * (y_pred_i - y_true_i)

    The factor 2 is real, not a convention — dropping it (as some texts do by
    defining MSE with a 1/2 out front) would make the analytical gradient
    disagree with a numerical check of *this* forward function.  We keep the
    forward and backward consistent instead.

    MSE penalises large errors quadratically, so it is sensitive to outliers.
    :class:`Huber` is the robust alternative.
    """

    name = "mse"

    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        _check_shapes(y_pred, y_true, "MeanSquaredError")
        p, t = y_pred.data, y_true.data
        n = len(p)
        return math.fsum((p[i] - t[i]) ** 2 for i in range(n)) / n

    def backward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        _check_shapes(y_pred, y_true, "MeanSquaredError")
        p, t = y_pred.data, y_true.data
        n = len(p)
        scale = 2.0 / n
        return Tensor([scale * (p[i] - t[i]) for i in range(n)], y_pred.shape)


class MeanAbsoluteError(Loss):
    """MAE / L1 loss.

    Forward::

        L = (1/N) * sum |y_pred - y_true|

    Backward::

        dL/dy_pred_i = sign(y_pred_i - y_true_i) / N

    The gradient has constant magnitude regardless of how wrong the
    prediction is, which is what makes MAE robust to outliers — a wildly
    wrong sample cannot dominate the update the way it can under MSE.  The
    cost is a kink at zero error (we take the subgradient 0 there) and a
    gradient that does not shrink as you converge.
    """

    name = "mae"

    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        _check_shapes(y_pred, y_true, "MeanAbsoluteError")
        p, t = y_pred.data, y_true.data
        n = len(p)
        return math.fsum(abs(p[i] - t[i]) for i in range(n)) / n

    def backward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        _check_shapes(y_pred, y_true, "MeanAbsoluteError")
        p, t = y_pred.data, y_true.data
        n = len(p)
        inv = 1.0 / n
        out = []
        for i in range(n):
            d = p[i] - t[i]
            out.append(inv * (1.0 if d > 0 else (-1.0 if d < 0 else 0.0)))
        return Tensor(out, y_pred.shape)


class BinaryCrossEntropy(Loss):
    """BCE — the loss for two-class problems with a sigmoid output.

    Forward, per element::

        L_i = -[ y*log(p) + (1-y)*log(1-p) ]

    and ``L`` is the mean over all elements.

    *Reading the formula.*  Only one of the two terms survives for a hard
    label.  If ``y = 1`` the loss is ``-log(p)``: zero when ``p = 1``, rising
    to infinity as ``p -> 0``.  If ``y = 0`` it is ``-log(1-p)``, mirrored.
    So the loss is the "surprise" of the true outcome under the predicted
    distribution, measured in nats.

    *Why not MSE for classification?*  Combined with sigmoid, MSE gives a
    gradient containing the factor ``sigma'(z)``, which is ~0 when the
    network is confidently wrong — precisely when you most need a large
    update.  BCE's ``log`` cancels that factor exactly (see below), so a
    confident mistake produces a correspondingly large gradient.

    Backward::

        dL/dp = (1/N) * (p - y) / (p * (1 - p))

    That denominator explodes as ``p`` nears 0 or 1, which is why ``p`` is
    clipped to ``[eps, 1-eps]``.  Note that when the previous layer is a
    sigmoid, its derivative is ``p(1-p)`` and multiplying the two gives::

        dL/dz = (1/N) * (p - y)

    — the same clean form as softmax + cross-entropy.  The clipping keeps
    the unfused path (which this class implements) numerically sane.
    """

    name = "binary_cross_entropy"

    def __init__(self, eps: float = 1e-7):
        # 1e-7 rather than 1e-12: it caps the per-sample loss near 16 nats and
        # the gradient near 1e7, keeping float64 arithmetic comfortable.
        self.eps = float(eps)

    def _validate_labels(self, y_true: Tensor) -> None:
        for v in y_true.data:
            if v < 0.0 or v > 1.0:
                raise ValueError(
                    f"BinaryCrossEntropy targets must lie in [0, 1], found {v}. "
                    f"If your labels are class indices like 2, you want "
                    f"CategoricalCrossEntropy with one-hot targets."
                )

    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        _check_shapes(y_pred, y_true, "BinaryCrossEntropy")
        self._validate_labels(y_true)
        e = self.eps
        p, t = y_pred.data, y_true.data
        n = len(p)
        total = 0.0
        for i in range(n):
            pi = p[i]
            # Clip away from the asymptotes so log() stays finite.
            if pi < e:
                pi = e
            elif pi > 1.0 - e:
                pi = 1.0 - e
            total += -(t[i] * math.log(pi) + (1.0 - t[i]) * math.log(1.0 - pi))
        return total / n

    def backward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        _check_shapes(y_pred, y_true, "BinaryCrossEntropy")
        e = self.eps
        p, t = y_pred.data, y_true.data
        n = len(p)
        inv = 1.0 / n
        out = [0.0] * n
        for i in range(n):
            pi = p[i]
            if pi < e:
                pi = e
            elif pi > 1.0 - e:
                pi = 1.0 - e
            out[i] = inv * (pi - t[i]) / (pi * (1.0 - pi))
        return Tensor(out, y_pred.shape)

    def config(self) -> Dict[str, object]:
        return {"eps": self.eps}


class CategoricalCrossEntropy(Loss):
    """Cross-entropy for mutually exclusive classes, with one-hot targets.

    Forward, per sample ``n`` over ``C`` classes::

        L_n = -sum over c of  y[n][c] * log(p[n][c])

    and ``L`` is the mean over the batch.  With a one-hot ``y``, all but one
    term is zero, so this reduces to ``-log(probability of the true class)``.

    Backward (the general, unfused form)::

        dL/dp[n][c] = -(1/N) * y[n][c] / p[n][c]

    **The fused softmax shortcut.**  When the layer before this loss is a
    softmax, composing the two gradients produces something remarkable.  With
    ``p = softmax(z)``::

        dL/dz[n][c] = sum over k of  dL/dp[n][k] * dp[n][k]/dz[n][c]

    Substituting the softmax Jacobian ``p_k(delta_kc - p_c)`` and using
    ``sum_k y_k = 1`` for a one-hot target, everything collapses to::

        dL/dz[n][c] = (1/N) * (p[n][c] - y[n][c])

    "predicted minus actual" — no division, no exponentials, nothing that can
    overflow.  :meth:`backward_fused` implements it, and
    :class:`~scratch_nn.model.Sequential` uses that path automatically when it
    sees a softmax output layer paired with this loss.  It is both faster and
    strictly more stable than the two-step route.
    """

    name = "categorical_cross_entropy"
    pairs_with_softmax = True

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        _check_shapes(y_pred, y_true, "CategoricalCrossEntropy")
        if y_pred.ndim != 2:
            raise TensorShapeError(
                f"CategoricalCrossEntropy expects (batch, classes), got {y_pred.shape}"
            )
        p, t = y_pred.data, y_true.data
        n_samples = y_pred.shape[0]
        total = 0.0
        for i in range(len(p)):
            if t[i] != 0.0:            # skip the zeros of a one-hot vector
                total += -t[i] * safe_log(p[i], self.eps)
        return total / n_samples

    def backward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """The general gradient, for a non-softmax predecessor."""
        _check_shapes(y_pred, y_true, "CategoricalCrossEntropy")
        p, t = y_pred.data, y_true.data
        n_samples = y_pred.shape[0]
        inv = 1.0 / n_samples
        e = self.eps
        out = [0.0] * len(p)
        for i in range(len(p)):
            if t[i] != 0.0:
                denom = p[i] if p[i] > e else e
                out[i] = -inv * t[i] / denom
        return Tensor(out, y_pred.shape)

    def backward_fused(self, softmax_output: Tensor, y_true: Tensor) -> Tensor:
        """``dL/dz = (p - y)/N`` — softmax and cross-entropy differentiated together."""
        _check_shapes(softmax_output, y_true, "CategoricalCrossEntropy")
        p, t = softmax_output.data, y_true.data
        inv = 1.0 / softmax_output.shape[0]
        return Tensor([inv * (p[i] - t[i]) for i in range(len(p))],
                      softmax_output.shape)

    def config(self) -> Dict[str, object]:
        return {"eps": self.eps}


class Huber(Loss):
    """Squared near zero, absolute far from it — MSE and MAE spliced together.

    Forward, with ``d = y_pred - y_true``::

        L = 0.5 * d^2                        if |d| <= delta
            delta * (|d| - 0.5*delta)        otherwise

    The two pieces are chosen so that both the value *and* the first
    derivative agree at ``|d| = delta``, making the loss smooth (C^1) there.

    Backward::

        dL/dy_pred = d                       if |d| <= delta
                     delta * sign(d)         otherwise

    So you get MSE's fine-grained gradient near the optimum and MAE's bounded
    gradient for outliers — the best of both.
    """

    name = "huber"

    def __init__(self, delta: float = 1.0):
        if delta <= 0:
            raise ValueError(f"Huber delta must be positive, got {delta}")
        self.delta = float(delta)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        _check_shapes(y_pred, y_true, "Huber")
        d = self.delta
        p, t = y_pred.data, y_true.data
        n = len(p)
        total = 0.0
        for i in range(n):
            err = p[i] - t[i]
            a = abs(err)
            total += 0.5 * err * err if a <= d else d * (a - 0.5 * d)
        return total / n

    def backward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        _check_shapes(y_pred, y_true, "Huber")
        d = self.delta
        p, t = y_pred.data, y_true.data
        n = len(p)
        inv = 1.0 / n
        out = [0.0] * n
        for i in range(n):
            err = p[i] - t[i]
            if abs(err) <= d:
                out[i] = inv * err
            else:
                out[i] = inv * d * (1.0 if err > 0 else -1.0)
        return Tensor(out, y_pred.shape)

    def config(self) -> Dict[str, object]:
        return {"delta": self.delta}


LOSSES: Dict[str, Type[Loss]] = {
    "mse": MeanSquaredError,
    "mean_squared_error": MeanSquaredError,
    "l2": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "mean_absolute_error": MeanAbsoluteError,
    "l1": MeanAbsoluteError,
    "bce": BinaryCrossEntropy,
    "binary_cross_entropy": BinaryCrossEntropy,
    "binary_crossentropy": BinaryCrossEntropy,
    "cce": CategoricalCrossEntropy,
    "categorical_cross_entropy": CategoricalCrossEntropy,
    "categorical_crossentropy": CategoricalCrossEntropy,
    "cross_entropy": CategoricalCrossEntropy,
    "huber": Huber,
}


def get_loss(spec, **kwargs) -> Loss:
    """Resolve ``"mse"`` / a class / an instance to a :class:`Loss` instance."""
    if isinstance(spec, Loss):
        return spec
    if isinstance(spec, type) and issubclass(spec, Loss):
        return spec(**kwargs)
    if isinstance(spec, str):
        key = spec.lower().strip()
        if key not in LOSSES:
            raise ValueError(
                f"unknown loss {spec!r}; available: {', '.join(sorted(set(LOSSES)))}"
            )
        return LOSSES[key](**kwargs)
    raise TypeError(f"cannot interpret {spec!r} as a loss")
