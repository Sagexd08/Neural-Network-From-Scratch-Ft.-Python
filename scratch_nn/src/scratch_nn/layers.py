"""
layers.py
=========

Layers, and the derivation of backpropagation through them.

The Dense layer, forward
------------------------
With ``X`` of shape ``(batch, in)``, ``W`` of shape ``(in, out)`` and ``b`` of
shape ``(out,)``::

    Z = X @ W + b          # (batch, out), b broadcast down the rows
    A = activation(Z)      # (batch, out)

Element-wise, for sample ``n`` and output unit ``j``::

    Z[n][j] = sum over i of X[n][i] * W[i][j]  +  b[j]

The Dense layer, backward
-------------------------
Backprop hands each layer ``dL/dA`` — how much the loss changes per unit
change in this layer's output — and asks for three things: ``dL/dW`` and
``dL/db`` (to update this layer), and ``dL/dX`` (to hand to the layer before).

**Step 1: through the activation.**  It is element-wise, so::

    dL/dZ[n][j] = dL/dA[n][j] * activation'(Z[n][j])

**Step 2: dL/dW.**  ``W[i][j]`` influences the loss through ``Z[n][j]`` for
every sample ``n``.  The multivariable chain rule says to sum those paths::

    dL/dW[i][j] = sum over n of  dL/dZ[n][j] * dZ[n][j]/dW[i][j]

and since ``dZ[n][j]/dW[i][j] = X[n][i]``::

    dL/dW[i][j] = sum over n of  X[n][i] * dL/dZ[n][j]

Recognise that sum: it is exactly the ``(i, j)`` entry of ``X^T @ dL/dZ``.
So::

    dW = X^T @ dZ                      # (in, batch) @ (batch, out) -> (in, out)

**Step 3: dL/db.**  ``dZ[n][j]/db[j] = 1``, so the paths just add up::

    db[j] = sum over n of dL/dZ[n][j]  # column sums of dZ -> (out,)

**Step 4: dL/dX.**  ``X[n][i]`` feeds every output unit ``j`` of sample ``n``::

    dL/dX[n][i] = sum over j of  dL/dZ[n][j] * dZ[n][j]/dX[n][i]
                = sum over j of  dL/dZ[n][j] * W[i][j]

which is the ``(n, i)`` entry of ``dZ @ W^T``::

    dX = dZ @ W^T                      # (batch, out) @ (out, in) -> (batch, in)

Those three lines are the entire Dense backward pass.  Note the pleasing
symmetry: forward multiplies by ``W``, backward multiplies by ``W^T``.

A note on batch averaging
-------------------------
``dW`` above is a *sum* over the batch, so its magnitude grows with batch
size — meaning you would have to retune the learning rate every time you
changed ``batch_size``.  To avoid that, the losses in this library divide
their gradient by the batch size, so ``dL/dA`` arrives already averaged and
everything downstream inherits the correct scale.  See ``losses.py``.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .activations import Activation, Identity, Softmax, get_activation
from .initialization import get_initializer, zeros_init
from .tensor import Tensor, TensorShapeError, zeros
from .utils import get_rng

__all__ = [
    "Layer",
    "Dense",
    "Dropout",
    "Flatten",
    "BatchNorm",
    "Activation_",
    "Residual",
]


class Layer:
    """Base class.

    Contract
    --------
    ``forward(x, training)``
        Compute the output and cache whatever ``backward`` needs.

    ``backward(grad_output)``
        Given ``dL/d(output)``, fill this layer's gradient buffers and return
        ``dL/d(input)``.

    ``parameters()`` / ``gradients()``
        Parallel lists.  ``parameters()[k]`` and ``gradients()[k]`` must refer
        to the same tensor pair, because optimizers zip them together.

    Gradients are *accumulated* into persistent buffers rather than
    reallocated, so :meth:`zero_grad` must be called before each backward
    pass.  Persistent buffers also let optimizers hold references across
    steps.
    """

    trainable: bool = True

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        raise NotImplementedError

    def backward(self, grad_output: Tensor) -> Tensor:
        raise NotImplementedError

    def parameters(self) -> List[Tensor]:
        """Learnable tensors, in a stable order."""
        return []

    def gradients(self) -> List[Tensor]:
        """Gradient buffers, aligned one-to-one with :meth:`parameters`."""
        return []

    def zero_grad(self) -> None:
        """Reset gradient buffers to zero before a new backward pass."""
        for g in self.gradients():
            g.zero_()

    def regularization_loss(self) -> float:
        """The penalty term this layer contributes to the total loss."""
        return 0.0

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Shape of the output given an input shape (batch axis excluded)."""
        return input_shape

    def config(self) -> Dict[str, Any]:
        """Everything needed to reconstruct this layer, minus the weights."""
        return {}

    def state_dict(self) -> Dict[str, Any]:
        """Learnable state, as JSON-friendly nested lists."""
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore from :meth:`state_dict` output."""
        return None

    def __call__(self, x: Tensor, training: bool = False) -> Tensor:
        return self.forward(x, training)

    def describe(self) -> str:
        """One-line summary for ``model.summary()``."""
        return type(self).__name__

    def __repr__(self) -> str:
        return self.describe()


class Dense(Layer):
    """A fully-connected layer: ``A = activation(X @ W + b)``.

    Parameters
    ----------
    input_size, output_size
        Number of input and output features.
    activation
        Name or instance; ``"relu"``, ``"sigmoid"``, ``"softmax"``, ...
    initialization
        Weight scheme — ``"he"`` (default), ``"xavier"``, ``"zeros"``, ...
    use_bias
        Set False to drop ``b`` (BatchNorm supplies its own shift).
    l1, l2
        Per-layer regularization strengths applied to ``W`` only, never to
        ``b``.  Penalising biases costs capacity and buys nothing, since a
        bias cannot cause overfitting the way a weight can.

    Example
    -------
    >>> layer = Dense(4, 8, activation="relu", initialization="he")
    >>> layer.forward(Tensor([[1.0, 2.0, 3.0, 4.0]])).shape
    (1, 8)
    """

    def __init__(self, input_size: int, output_size: int, activation="identity",
                 initialization="he", use_bias: bool = True,
                 l1: float = 0.0, l2: float = 0.0,
                 rng: random.Random | None = None):
        if input_size <= 0 or output_size <= 0:
            raise ValueError(
                f"Dense sizes must be positive, got input_size={input_size}, "
                f"output_size={output_size}"
            )
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.use_bias = bool(use_bias)
        self.l1 = float(l1)
        self.l2 = float(l2)

        self.activation_name = (
            activation if isinstance(activation, str)
            else getattr(activation, "name", type(activation).__name__.lower())
        )
        self.activation: Activation = get_activation(activation)
        self.initialization = initialization if isinstance(initialization, str) else "custom"

        # W[i][j] is the weight from input i to output unit j.
        init_fn = get_initializer(initialization)
        self.W: Tensor = init_fn(self.input_size, self.output_size, rng or get_rng())
        # Biases start at zero: they are already made distinct by the weights
        # feeding each unit, so no symmetry breaking is needed here.
        self.b: Tensor = zeros((self.output_size,)) if use_bias else None  # type: ignore

        # Persistent gradient buffers, same shapes as the parameters.
        self.dW: Tensor = zeros(self.W.shape)
        self.db: Tensor = zeros((self.output_size,)) if use_bias else None  # type: ignore

        # Cached forward values needed by backward.
        self._input: Optional[Tensor] = None
        self._pre_activation: Optional[Tensor] = None

    # -- forward ------------------------------------------------------------

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        """``Z = X @ W + b``, then ``A = activation(Z)``."""
        if x.ndim != 2:
            raise TensorShapeError(
                f"Dense expects a 2-D (batch, features) input, got shape {x.shape}. "
                f"Add a Flatten layer if your data has more axes."
            )
        if x.shape[1] != self.input_size:
            raise TensorShapeError(
                f"Dense(input_size={self.input_size}) received input with "
                f"{x.shape[1]} features (shape {x.shape})"
            )

        # Z = X @ W  -- the matrix product that is the layer's whole linear part
        z = x.matmul(self.W)

        # + b, broadcast across the batch: every row gets the same bias vector.
        if self.use_bias:
            zd, bd = z.data, self.b.data
            cols = self.output_size
            for n in range(z.shape[0]):
                base = n * cols
                for j in range(cols):
                    zd[base + j] += bd[j]

        self._input = x
        self._pre_activation = z
        return self.activation.forward(z)

    # -- backward -----------------------------------------------------------

    def backward(self, grad_output: Tensor, fused_activation_grad: bool = False) -> Tensor:
        """Compute ``dW``, ``db`` and return ``dX``.

        ``fused_activation_grad=True`` means the caller has already computed
        ``dL/dZ`` (skipping the activation derivative).  This is how the
        softmax + categorical-cross-entropy shortcut ``dZ = p - y`` is
        plumbed through — see :mod:`scratch_nn.model`.
        """
        if self._input is None or self._pre_activation is None:
            raise RuntimeError("Dense.backward called before forward")

        x = self._input
        batch = x.shape[0]

        # --- Step 1: push the gradient back through the activation ---------
        #     dL/dZ = dL/dA * activation'(Z)
        dz = grad_output if fused_activation_grad else self.activation.backward(grad_output)

        if dz.shape != (batch, self.output_size):
            raise TensorShapeError(
                f"Dense.backward expected gradient of shape "
                f"{(batch, self.output_size)}, got {dz.shape}"
            )

        # --- Step 2: dW = X^T @ dZ ----------------------------------------
        # Written as an explicit triple loop rather than x.T.matmul(dz) so
        # that no (in, batch) transpose is materialised, and so the sum over
        # the batch axis is visible in the code.
        xd, dzd, dwd = x.data, dz.data, self.dW.data
        in_size, out_size = self.input_size, self.output_size
        for i in range(in_size):
            row_base = i          # X^T[i][n] == X[n][i], i.e. flat index n*in + i
            grad_base = i * out_size
            for j in range(out_size):
                acc = 0.0
                for n in range(batch):
                    acc += xd[n * in_size + row_base] * dzd[n * out_size + j]
                dwd[grad_base + j] += acc      # += so gradients accumulate

        # --- Step 3: db = column sums of dZ -------------------------------
        if self.use_bias:
            dbd = self.db.data
            for j in range(out_size):
                acc = 0.0
                for n in range(batch):
                    acc += dzd[n * out_size + j]
                dbd[j] += acc

        # --- Regularization contributes directly to the weight gradient ----
        #     d/dW of (l2 * sum W^2)  = 2*l2*W   -> conventionally lambda*W
        #     d/dW of (l1 * sum |W|)  = l1*sign(W)
        if self.l2 or self.l1:
            wd = self.W.data
            l1, l2 = self.l1, self.l2
            for k in range(len(wd)):
                w = wd[k]
                if l2:
                    dwd[k] += 2.0 * l2 * w
                if l1:
                    dwd[k] += l1 * (1.0 if w > 0 else (-1.0 if w < 0 else 0.0))

        # --- Step 4: dX = dZ @ W^T ----------------------------------------
        wd = self.W.data
        dx = [0.0] * (batch * in_size)
        for n in range(batch):
            dz_base = n * out_size
            dx_base = n * in_size
            for i in range(in_size):
                w_base = i * out_size
                acc = 0.0
                for j in range(out_size):
                    acc += dzd[dz_base + j] * wd[w_base + j]
                dx[dx_base + i] = acc
        return Tensor(dx, (batch, in_size))

    # -- plumbing -----------------------------------------------------------

    def parameters(self) -> List[Tensor]:
        return [self.W, self.b] if self.use_bias else [self.W]

    def gradients(self) -> List[Tensor]:
        return [self.dW, self.db] if self.use_bias else [self.dW]

    def regularization_loss(self) -> float:
        """``l2 * sum(W^2) + l1 * sum(|W|)`` — the penalty added to the loss."""
        if not (self.l1 or self.l2):
            return 0.0
        total = 0.0
        if self.l2:
            total += self.l2 * math.fsum(w * w for w in self.W.data)
        if self.l1:
            total += self.l1 * math.fsum(abs(w) for w in self.W.data)
        return total

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return (self.output_size,)

    def config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "activation": self.activation_name,
            "initialization": self.initialization,
            "use_bias": self.use_bias,
            "l1": self.l1,
            "l2": self.l2,
        }
        act_cfg = self.activation.config()
        if act_cfg:
            cfg["activation_config"] = act_cfg
        return cfg

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"W": self.W.tolist()}
        if self.use_bias:
            state["b"] = self.b.tolist()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        w = Tensor(state["W"])
        if w.shape != self.W.shape:
            raise TensorShapeError(
                f"saved W has shape {w.shape}, this layer expects {self.W.shape}"
            )
        self.W = w
        self.dW = zeros(w.shape)
        if self.use_bias:
            b = Tensor(state["b"])
            if b.shape != (self.output_size,):
                raise TensorShapeError(
                    f"saved b has shape {b.shape}, expected {(self.output_size,)}"
                )
            self.b = b
            self.db = zeros((self.output_size,))

    def describe(self) -> str:
        n_params = self.W.size + (self.output_size if self.use_bias else 0)
        reg = ""
        if self.l1 or self.l2:
            reg = f", l1={self.l1:g}, l2={self.l2:g}"
        return (f"Dense({self.input_size} -> {self.output_size}, "
                f"activation={self.activation_name}, init={self.initialization}{reg}) "
                f"[{n_params} params]")


class Dropout(Layer):
    """Randomly zero a fraction of activations during training.

    Forward (training)::

        mask_i ~ Bernoulli(1 - rate)
        a_i = x_i * mask_i / (1 - rate)

    Forward (inference)::

        a_i = x_i

    Dropout stops units from co-adapting: since any given input may vanish on
    any given step, no unit can rely on a specific partner being present, so
    the network is pushed toward redundant, distributed representations.

    The ``/(1 - rate)`` is *inverted dropout*.  Without it, a unit sees an
    expected input of ``(1-rate)*x`` while training but the full ``x`` at
    inference — a systematic scale mismatch.  Scaling up during training
    keeps the expected value at ``x`` in both regimes, so inference needs no
    correction at all.

    Backward: the same mask (and the same scale) applied to the incoming
    gradient — a dropped unit contributed nothing, so it receives nothing.
    """

    trainable = False

    def __init__(self, rate: float = 0.5, rng: random.Random | None = None):
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"dropout rate must be in [0, 1), got {rate}")
        self.rate = float(rate)
        self.scale = 1.0 / (1.0 - self.rate) if self.rate > 0 else 1.0
        self._rng = rng
        self._mask: Optional[List[float]] = None

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        # Inference (and rate==0) is the identity: this is the single most
        # common dropout bug, so the check is explicit and first.
        if not training or self.rate == 0.0:
            self._mask = None
            return x
        r = self._rng or get_rng()
        keep = 1.0 - self.rate
        scale = self.scale
        mask = [scale if r.random() < keep else 0.0 for _ in range(x.size)]
        self._mask = mask
        d = x.data
        return Tensor([d[i] * mask[i] for i in range(len(d))], x.shape)

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._mask is None:
            return grad_output
        g, m = grad_output.data, self._mask
        return Tensor([g[i] * m[i] for i in range(len(g))], grad_output.shape)

    def config(self) -> Dict[str, Any]:
        return {"rate": self.rate}

    def describe(self) -> str:
        return f"Dropout(rate={self.rate})"


class Flatten(Layer):
    """Collapse everything after the batch axis into one feature axis.

    ``(batch, 2, 3)`` becomes ``(batch, 6)``.  Backward simply restores the
    original shape — no arithmetic, since no values changed.
    """

    trainable = False

    def __init__(self):
        self._input_shape: Optional[Tuple[int, ...]] = None

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        if x.ndim < 2:
            raise TensorShapeError(
                f"Flatten expects at least 2 axes (batch first), got shape {x.shape}"
            )
        self._input_shape = x.shape
        features = 1
        for d in x.shape[1:]:
            features *= d
        return x.reshape(x.shape[0], features)

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._input_shape is None:
            raise RuntimeError("Flatten.backward called before forward")
        return grad_output.reshape(self._input_shape)

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        total = 1
        for d in input_shape:
            total *= d
        return (total,)

    def describe(self) -> str:
        return "Flatten()"


class Activation_(Layer):
    """A standalone activation, for when you want it as its own layer.

    Named with a trailing underscore to avoid colliding with
    :class:`scratch_nn.activations.Activation`.
    """

    trainable = False

    def __init__(self, activation="relu"):
        self.activation_name = (
            activation if isinstance(activation, str)
            else getattr(activation, "name", type(activation).__name__.lower())
        )
        self.activation = get_activation(activation)

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        return self.activation.forward(x)

    def backward(self, grad_output: Tensor) -> Tensor:
        return self.activation.backward(grad_output)

    def config(self) -> Dict[str, Any]:
        return {"activation": self.activation_name}

    def describe(self) -> str:
        return f"Activation({self.activation_name})"


class BatchNorm(Layer):
    """Normalise each feature across the batch, then re-scale and re-shift.

    Forward (training), per feature ``j`` over a batch of ``N``::

        mu_j    = (1/N) * sum_n x[n][j]
        var_j   = (1/N) * sum_n (x[n][j] - mu_j)^2
        xhat[n][j] = (x[n][j] - mu_j) / sqrt(var_j + eps)
        y[n][j]    = gamma_j * xhat[n][j] + beta_j

    Forward (inference) uses running averages of ``mu`` and ``var`` collected
    during training, so a single sample can be predicted without a batch.

    ``gamma`` and ``beta`` are learnable: normalizing to zero-mean/unit-
    variance would otherwise strip the layer of expressiveness (it could not,
    for instance, exploit sigmoid's saturating region even when useful).
    With them, the network can undo the normalization if that is optimal.

    Backward.  This is the fiddliest derivation in the file, because ``mu``
    and ``var`` both depend on *every* sample, so each ``x[n][j]`` influences
    the output of every other sample in the batch.  Working through the three
    paths (direct, via ``mu``, via ``var``) and collecting terms gives the
    standard closed form::

        dxhat  = dy * gamma
        dvar   = sum_n dxhat[n] * (x[n]-mu) * (-1/2) * (var+eps)^(-3/2)
        dmu    = sum_n dxhat[n] * (-1/sqrt(var+eps))  +  dvar * mean(-2*(x-mu))
        dx[n]  = dxhat[n]/sqrt(var+eps) + dvar*2*(x[n]-mu)/N + dmu/N

    which we implement below in the algebraically-simplified form::

        dx[n] = (1/(N*std)) * (N*dxhat[n] - sum(dxhat) - xhat[n]*sum(dxhat*xhat))
    """

    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        self.num_features = int(num_features)
        self.momentum = float(momentum)
        self.eps = float(eps)

        self.gamma = Tensor([1.0] * self.num_features, (self.num_features,))
        self.beta = zeros((self.num_features,))
        self.dgamma = zeros((self.num_features,))
        self.dbeta = zeros((self.num_features,))

        # Buffers, not parameters: updated by observation, not by gradient.
        self.running_mean = zeros((self.num_features,))
        self.running_var = Tensor([1.0] * self.num_features, (self.num_features,))

        self._xhat: Optional[List[float]] = None
        self._std: Optional[List[float]] = None
        self._batch: int = 0

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise TensorShapeError(
                f"BatchNorm({self.num_features}) expects shape (batch, "
                f"{self.num_features}), got {x.shape}"
            )
        n, f = x.shape
        xd = x.data
        out = [0.0] * len(xd)

        if training:
            if n < 2:
                raise ValueError(
                    "BatchNorm needs at least 2 samples per batch during training "
                    f"(got {n}); the batch variance of a single sample is 0"
                )
            mean = [0.0] * f
            var = [0.0] * f
            for j in range(f):
                s = 0.0
                for i in range(n):
                    s += xd[i * f + j]
                m = s / n
                mean[j] = m
                v = 0.0
                for i in range(n):
                    d = xd[i * f + j] - m
                    v += d * d
                var[j] = v / n

            std = [math.sqrt(var[j] + self.eps) for j in range(f)]
            xhat = [0.0] * len(xd)
            gd, bd = self.gamma.data, self.beta.data
            for i in range(n):
                base = i * f
                for j in range(f):
                    h = (xd[base + j] - mean[j]) / std[j]
                    xhat[base + j] = h
                    out[base + j] = gd[j] * h + bd[j]

            # Running statistics for inference: an exponential moving average.
            rm, rv, mom = self.running_mean.data, self.running_var.data, self.momentum
            for j in range(f):
                rm[j] = mom * rm[j] + (1.0 - mom) * mean[j]
                rv[j] = mom * rv[j] + (1.0 - mom) * var[j]

            self._xhat = xhat
            self._std = std
            self._batch = n
        else:
            rm, rv = self.running_mean.data, self.running_var.data
            gd, bd = self.gamma.data, self.beta.data
            inv = [1.0 / math.sqrt(rv[j] + self.eps) for j in range(f)]
            for i in range(n):
                base = i * f
                for j in range(f):
                    out[base + j] = gd[j] * (xd[base + j] - rm[j]) * inv[j] + bd[j]
            self._xhat = None
        return Tensor(out, x.shape)

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._xhat is None or self._std is None:
            raise RuntimeError("BatchNorm.backward called before a training forward pass")
        n, f = self._batch, self.num_features
        gy, xhat, std = grad_output.data, self._xhat, self._std
        gd = self.gamma.data
        dgd, dbd = self.dgamma.data, self.dbeta.data
        dx = [0.0] * len(gy)

        for j in range(f):
            # Per-feature accumulators for the three terms of the formula.
            sum_dy = 0.0        # sum_n dxhat[n]
            sum_dy_xhat = 0.0   # sum_n dxhat[n]*xhat[n]
            for i in range(n):
                k = i * f + j
                dxhat = gy[k] * gd[j]
                sum_dy += dxhat
                sum_dy_xhat += dxhat * xhat[k]
                # gamma and beta gradients: straightforward chain rule.
                dgd[j] += gy[k] * xhat[k]
                dbd[j] += gy[k]
            inv = 1.0 / (n * std[j])
            for i in range(n):
                k = i * f + j
                dxhat = gy[k] * gd[j]
                dx[k] = inv * (n * dxhat - sum_dy - xhat[k] * sum_dy_xhat)
        return Tensor(dx, grad_output.shape)

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]

    def gradients(self) -> List[Tensor]:
        return [self.dgamma, self.dbeta]

    def config(self) -> Dict[str, Any]:
        return {"num_features": self.num_features, "momentum": self.momentum,
                "eps": self.eps}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "gamma": self.gamma.tolist(),
            "beta": self.beta.tolist(),
            "running_mean": self.running_mean.tolist(),
            "running_var": self.running_var.tolist(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.gamma = Tensor(state["gamma"])
        self.beta = Tensor(state["beta"])
        self.running_mean = Tensor(state["running_mean"])
        self.running_var = Tensor(state["running_var"])
        self.dgamma = zeros(self.gamma.shape)
        self.dbeta = zeros(self.beta.shape)

    def describe(self) -> str:
        return f"BatchNorm({self.num_features}) [{2 * self.num_features} params]"


class Residual(Layer):
    """A skip connection: ``y = x + block(x)``.

    The gradient of the sum is the sum of gradients, so::

        dL/dx = dL/dy + block_backward(dL/dy)

    That leading ``dL/dy`` is the point: it is an unobstructed path for the
    gradient to reach earlier layers, no matter how small the block's own
    gradient becomes.  It is what makes very deep networks trainable.

    The block must preserve the feature width, since ``x`` and ``block(x)``
    are added element-wise.
    """

    def __init__(self, layers: Sequence[Layer]):
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("Residual needs at least one inner layer")

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer.forward(out, training)
        if out.shape != x.shape:
            raise TensorShapeError(
                f"Residual block changed the shape from {x.shape} to {out.shape}; "
                f"a skip connection requires them to match"
            )
        return out.add(x)

    def backward(self, grad_output: Tensor) -> Tensor:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        # The identity branch passes grad_output through unchanged.
        return grad.add(grad_output)

    def parameters(self) -> List[Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]

    def gradients(self) -> List[Tensor]:
        return [g for layer in self.layers for g in layer.gradients()]

    def regularization_loss(self) -> float:
        return sum(layer.regularization_loss() for layer in self.layers)

    def config(self) -> Dict[str, Any]:
        return {"layers": [{"type": type(l).__name__, "config": l.config()}
                           for l in self.layers]}

    def state_dict(self) -> Dict[str, Any]:
        return {"layers": [l.state_dict() for l in self.layers]}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for layer, sub in zip(self.layers, state["layers"]):
            layer.load_state_dict(sub)

    def describe(self) -> str:
        inner = " -> ".join(l.describe() for l in self.layers)
        return f"Residual({inner})"
