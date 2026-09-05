"""
activations.py
==============

Nonlinearities and their derivatives.

Why any of this exists
----------------------
A Dense layer computes ``Z = XW + b``, which is an *affine* function.  Stack
two of them and you get::

    (XW1 + b1)W2 + b2  =  X(W1 W2) + (b1 W2 + b2)  =  XW' + b'

— still affine.  Without a nonlinearity between layers, a 50-layer network
has exactly the representational power of a single layer, and XOR (which is
not linearly separable) stays unsolvable forever.  The activation function is
what breaks that collapse.

The API
-------
Each activation is a class with two methods:

``forward(x)``
    Apply the function element-wise, caching whatever the backward pass needs.

``backward(grad_output)``
    Given ``dL/da`` (the gradient of the loss w.r.t. this layer's *output*),
    return ``dL/dz`` (w.r.t. its *input*).  For an element-wise function that
    is just the chain rule::

        dL/dz_i = dL/da_i * f'(z_i)

Softmax is the exception — its Jacobian is not diagonal, because output ``i``
depends on *every* input.  See the class docstring for how that is handled.
"""

from __future__ import annotations

import math
from typing import Dict, Type

from .tensor import Tensor
from .utils import safe_exp

__all__ = [
    "Activation",
    "Identity",
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "ELU",
    "GELU",
    "Swish",
    "get_activation",
    "ACTIVATIONS",
]


class Activation:
    """Base class for element-wise nonlinearities."""

    name: str = "activation"

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``f(x)`` and cache what ``backward`` will need."""
        raise NotImplementedError

    def backward(self, grad_output: Tensor) -> Tensor:
        """Turn ``dL/d(output)`` into ``dL/d(input)``."""
        raise NotImplementedError

    def config(self) -> Dict[str, object]:
        """Constructor arguments, for serialization."""
        return {}

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        cfg = self.config()
        args = ", ".join(f"{k}={v}" for k, v in cfg.items())
        return f"{type(self).__name__}({args})"


class Identity(Activation):
    """``f(x) = x``, derivative 1.

    Used for regression output layers, where squashing the output into a
    bounded range would make it impossible to predict values outside it.
    """

    name = "identity"

    def forward(self, x: Tensor) -> Tensor:
        return x

    def backward(self, grad_output: Tensor) -> Tensor:
        # d/dx of x is 1, so the gradient passes through untouched.
        return grad_output


class ReLU(Activation):
    """Rectified Linear Unit.

    Forward::

        f(x) = max(0, x)

    Derivative::

        f'(x) = 1  if x > 0
                0  if x < 0

    At exactly x = 0 the function has a kink and is not differentiable.  We
    define the subgradient there as 0 (the standard convention).  Since a
    float landing on exactly 0.0 is vanishingly rare, the choice does not
    matter in practice.

    ReLU is the default hidden activation because its gradient is exactly 1
    over the whole positive half-line.  Sigmoid and tanh have derivatives
    bounded by 0.25 and 1.0 that *decay towards zero* as inputs grow, so
    multiplying them layer after layer drives gradients to zero — the
    "vanishing gradient" problem.  ReLU's constant 1 avoids that entirely.

    Its cost is the "dying ReLU": a unit pushed permanently negative has zero
    gradient forever and can never recover.  :class:`LeakyReLU` fixes that.
    """

    name = "relu"

    def __init__(self):
        self._mask: list[float] | None = None
        self._shape: tuple[int, ...] | None = None

    def forward(self, x: Tensor) -> Tensor:
        data = x.data
        out = [0.0] * len(data)
        # Cache the derivative itself rather than the inputs: it is the same
        # amount of memory but saves recomputing the comparison in backward.
        mask = [0.0] * len(data)
        for i in range(len(data)):
            v = data[i]
            if v > 0.0:
                out[i] = v
                mask[i] = 1.0
        self._mask = mask
        self._shape = x.shape
        return Tensor(out, x.shape)

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._mask is None:
            raise RuntimeError("ReLU.backward called before forward")
        if grad_output.shape != self._shape:
            raise ValueError(
                f"ReLU.backward got gradient of shape {grad_output.shape}, but the "
                f"forward pass produced {self._shape}"
            )
        g, m = grad_output.data, self._mask
        # dL/dx = dL/da * f'(x): the gradient is simply zeroed where x <= 0.
        return Tensor([g[i] * m[i] for i in range(len(g))], grad_output.shape)


class LeakyReLU(Activation):
    """ReLU with a small slope on the negative side.

    Forward::

        f(x) = x       if x > 0
               alpha*x otherwise

    Derivative::

        f'(x) = 1      if x > 0
                alpha  otherwise

    With ``alpha = 0.01`` a negative unit still receives 1% of the gradient,
    so it can climb back to positive territory instead of dying permanently.
    """

    name = "leaky_relu"

    def __init__(self, alpha: float = 0.01):
        if alpha < 0:
            raise ValueError(f"LeakyReLU alpha must be non-negative, got {alpha}")
        self.alpha = float(alpha)
        self._mask: list[float] | None = None
        self._shape: tuple[int, ...] | None = None

    def forward(self, x: Tensor) -> Tensor:
        a = self.alpha
        data = x.data
        out = [0.0] * len(data)
        mask = [0.0] * len(data)
        for i in range(len(data)):
            v = data[i]
            if v > 0.0:
                out[i] = v
                mask[i] = 1.0
            else:
                out[i] = a * v
                mask[i] = a
        self._mask = mask
        self._shape = x.shape
        return Tensor(out, x.shape)

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._mask is None:
            raise RuntimeError("LeakyReLU.backward called before forward")
        g, m = grad_output.data, self._mask
        return Tensor([g[i] * m[i] for i in range(len(g))], grad_output.shape)

    def config(self) -> Dict[str, object]:
        return {"alpha": self.alpha}


class Sigmoid(Activation):
    """Logistic sigmoid, squashing any real number into ``(0, 1)``.

    Forward::

        s(x) = 1 / (1 + e^-x)

    Derivative — one of the tidiest in all of machine learning::

        s'(x) = s(x) * (1 - s(x))

    *Why that derivative holds.*  Write ``s = (1 + e^-x)^-1``.  By the chain
    rule, ``ds/dx = -(1 + e^-x)^-2 * (-e^-x) = e^-x / (1 + e^-x)^2``.  Split
    that product: ``= [1/(1+e^-x)] * [e^-x/(1+e^-x)] = s * (1 - s)``, because
    ``e^-x/(1+e^-x) = 1 - 1/(1+e^-x)``.  So the backward pass needs only the
    cached *output* — no exponentials at all.

    *Numerical stability.*  The naive formula computes ``e^-x``, which
    overflows for x around -745.  The standard fix branches on the sign::

        x >= 0:  s = 1 / (1 + e^-x)        # e^-x <= 1, safe
        x <  0:  s = e^x / (1 + e^x)       # e^x  <= 1, safe

    Both branches only ever exponentiate a non-positive number, so overflow
    is structurally impossible.
    """

    name = "sigmoid"

    def __init__(self):
        self._output: Tensor | None = None

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0.0:
            return 1.0 / (1.0 + safe_exp(-x))
        e = safe_exp(x)
        return e / (1.0 + e)

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor([self._sigmoid(v) for v in x.data], x.shape)
        self._output = out  # cached because s' is expressed in terms of s
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._output is None:
            raise RuntimeError("Sigmoid.backward called before forward")
        g, s = grad_output.data, self._output.data
        return Tensor([g[i] * s[i] * (1.0 - s[i]) for i in range(len(g))],
                      grad_output.shape)


class Tanh(Activation):
    """Hyperbolic tangent, squashing into ``(-1, 1)``.

    Forward::

        tanh(x) = (e^x - e^-x) / (e^x + e^-x)

    Derivative::

        tanh'(x) = 1 - tanh(x)^2

    Tanh is zero-centred, unlike sigmoid whose outputs are all positive.  A
    layer fed only-positive inputs has all its weight gradients sharing one
    sign, which makes the optimizer zig-zag; tanh avoids that, and so is the
    better choice of the two saturating activations for hidden layers.

    ``math.tanh`` is already implemented stably in C, so we call it directly.
    """

    name = "tanh"

    def __init__(self):
        self._output: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor([math.tanh(v) for v in x.data], x.shape)
        self._output = out
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._output is None:
            raise RuntimeError("Tanh.backward called before forward")
        g, t = grad_output.data, self._output.data
        return Tensor([g[i] * (1.0 - t[i] * t[i]) for i in range(len(g))],
                      grad_output.shape)


class Softmax(Activation):
    """Turn a row of scores ("logits") into a probability distribution.

    Forward, applied independently to each row of a ``(batch, classes)``
    tensor::

        p_i = e^(z_i) / sum_j e^(z_j)

    Every ``p_i`` is positive and each row sums to exactly 1.

    *Numerical stability.*  Logits of 1000 would make ``e^z`` overflow to
    ``inf``, and ``inf/inf`` is ``nan``.  The fix uses the fact that softmax
    is invariant to shifting all inputs by a constant::

        e^(z_i - c) / sum_j e^(z_j - c)
            = [e^-c * e^(z_i)] / [e^-c * sum_j e^(z_j)]
            = e^(z_i) / sum_j e^(z_j)

    Choosing ``c = max(z)`` makes the largest exponent exactly ``e^0 = 1``,
    so nothing can overflow and the denominator is at least 1.

    *The backward pass.*  Unlike every other activation here, softmax is not
    element-wise: ``p_i`` depends on all of ``z``.  Its Jacobian is::

        dp_i/dz_j = p_i * (delta_ij - p_j)

    so the vector-Jacobian product a backward pass needs is::

        dL/dz_i = p_i * ( dL/dp_i - sum_j dL/dp_j * p_j )

    The bracketed sum is a per-row dot product, so the whole thing costs
    O(classes) per row, not O(classes^2) — we never build the Jacobian.

    In practice you should pair softmax with
    :class:`~scratch_nn.losses.CategoricalCrossEntropy`, whose combined
    gradient collapses to the beautifully simple ``p - y``.  That path skips
    this method entirely (see ``Dense.backward``'s ``fused`` handling), which
    is both faster and more numerically stable.
    """

    name = "softmax"

    def __init__(self):
        self._output: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Softmax expects a 2-D (batch, classes) tensor, got shape {x.shape}"
            )
        rows, cols = x.shape
        data = x.data
        out = [0.0] * len(data)
        for r in range(rows):
            base = r * cols
            # 1. shift by the row max so the biggest exponent is e^0 = 1
            row_max = data[base]
            for j in range(1, cols):
                if data[base + j] > row_max:
                    row_max = data[base + j]
            # 2. exponentiate the shifted scores and accumulate the sum
            total = 0.0
            for j in range(cols):
                e = safe_exp(data[base + j] - row_max)
                out[base + j] = e
                total += e
            # 3. normalise; total >= 1 always, since one term is exactly e^0
            for j in range(cols):
                out[base + j] /= total
        result = Tensor(out, x.shape)
        self._output = result
        return result

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._output is None:
            raise RuntimeError("Softmax.backward called before forward")
        p = self._output
        rows, cols = p.shape
        pd, gd = p.data, grad_output.data
        out = [0.0] * len(pd)
        for r in range(rows):
            base = r * cols
            # The shared per-row term: sum_j (dL/dp_j * p_j)
            row_dot = 0.0
            for j in range(cols):
                row_dot += gd[base + j] * pd[base + j]
            # dL/dz_i = p_i * (dL/dp_i - row_dot)
            for i in range(cols):
                out[base + i] = pd[base + i] * (gd[base + i] - row_dot)
        return Tensor(out, p.shape)


class ELU(Activation):
    """Exponential Linear Unit.

    Forward::

        f(x) = x                  if x > 0
               alpha*(e^x - 1)    otherwise

    Derivative::

        f'(x) = 1                 if x > 0
                f(x) + alpha      otherwise   [ = alpha*e^x ]

    Like LeakyReLU it keeps negative units alive, but it saturates to
    ``-alpha`` rather than growing without bound, which makes it more robust
    to large negative inputs.  Mean activations sit closer to zero than
    ReLU's, which helps the next layer's conditioning.
    """

    name = "elu"

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self._input: Tensor | None = None
        self._output: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        a = self.alpha
        out = [v if v > 0.0 else a * (safe_exp(v) - 1.0) for v in x.data]
        self._input = x
        result = Tensor(out, x.shape)
        self._output = result
        return result

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._input is None or self._output is None:
            raise RuntimeError("ELU.backward called before forward")
        xs, ys, g = self._input.data, self._output.data, grad_output.data
        a = self.alpha
        return Tensor(
            [g[i] * (1.0 if xs[i] > 0.0 else ys[i] + a) for i in range(len(g))],
            grad_output.shape,
        )

    def config(self) -> Dict[str, object]:
        return {"alpha": self.alpha}


class GELU(Activation):
    """Gaussian Error Linear Unit (the tanh approximation used by GPT/BERT).

    Forward::

        f(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))

    Intuition: instead of ReLU's hard gate (keep x, or zero it), GELU weights
    x by the probability that a standard normal variable is below x — a soft,
    smooth gate.  Being smooth everywhere (no kink at 0) tends to help deep
    networks optimize.

    Derivative, by the product and chain rules with ``u = c(x + kx^3)``::

        f'(x) = 0.5*(1 + tanh(u)) + 0.5*x*(1 - tanh(u)^2)*c*(1 + 3k x^2)
    """

    name = "gelu"
    _C = math.sqrt(2.0 / math.pi)
    _K = 0.044715

    def __init__(self):
        self._input: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        c, k = self._C, self._K
        out = [0.5 * v * (1.0 + math.tanh(c * (v + k * v * v * v))) for v in x.data]
        self._input = x
        return Tensor(out, x.shape)

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._input is None:
            raise RuntimeError("GELU.backward called before forward")
        c, k = self._C, self._K
        xs, g = self._input.data, grad_output.data
        out = [0.0] * len(g)
        for i in range(len(g)):
            v = xs[i]
            u = c * (v + k * v * v * v)
            t = math.tanh(u)
            du = c * (1.0 + 3.0 * k * v * v)
            out[i] = g[i] * (0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * du)
        return Tensor(out, grad_output.shape)


class Swish(Activation):
    """Swish / SiLU: ``f(x) = x * sigmoid(beta*x)``.

    Derivative, by the product rule with ``s = sigmoid(beta*x)``::

        f'(x) = s + beta*x*s*(1 - s)
              = beta*f(x) + s*(1 - beta*f(x))

    Self-gating like GELU, smooth everywhere, and slightly cheaper to compute.
    """

    name = "swish"

    def __init__(self, beta: float = 1.0):
        self.beta = float(beta)
        self._input: Tensor | None = None
        self._sig: list[float] | None = None

    def forward(self, x: Tensor) -> Tensor:
        b = self.beta
        sig = [Sigmoid._sigmoid(b * v) for v in x.data]
        self._input = x
        self._sig = sig
        return Tensor([x.data[i] * sig[i] for i in range(len(sig))], x.shape)

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._input is None or self._sig is None:
            raise RuntimeError("Swish.backward called before forward")
        b = self.beta
        xs, s, g = self._input.data, self._sig, grad_output.data
        return Tensor(
            [g[i] * (s[i] + b * xs[i] * s[i] * (1.0 - s[i])) for i in range(len(g))],
            grad_output.shape,
        )

    def config(self) -> Dict[str, object]:
        return {"beta": self.beta}


# Registry so layers can be built from strings ("relu") as well as objects.
ACTIVATIONS: Dict[str, Type[Activation]] = {
    "identity": Identity,
    "linear": Identity,
    "none": Identity,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "leakyrelu": LeakyReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
    "elu": ELU,
    "gelu": GELU,
    "swish": Swish,
    "silu": Swish,
}


def get_activation(spec, **kwargs) -> Activation:
    """Resolve ``"relu"``, ``ReLU``, or an ``Activation`` instance to an instance.

    A fresh instance is always returned for string/class inputs, because each
    activation caches per-batch state and so cannot be shared between layers.
    """
    if spec is None:
        return Identity()
    if isinstance(spec, Activation):
        return spec
    if isinstance(spec, type) and issubclass(spec, Activation):
        return spec(**kwargs)
    if isinstance(spec, str):
        key = spec.lower().strip()
        if key not in ACTIVATIONS:
            raise ValueError(
                f"unknown activation {spec!r}; available: "
                f"{', '.join(sorted(set(ACTIVATIONS)))}"
            )
        return ACTIVATIONS[key](**kwargs)
    raise TypeError(f"cannot interpret {spec!r} as an activation")
