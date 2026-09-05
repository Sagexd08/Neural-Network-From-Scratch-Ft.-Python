"""
optimizers.py
=============

Update rules: given a parameter and its gradient, decide the next value.

The gradient ``dL/dW`` points in the direction of steepest *increase* of the
loss, so every rule below is some refinement of "step the other way".

    SGD           W <- W - lr * g
    Momentum      accumulate a velocity, step along that
    Adam          per-parameter learning rates from gradient statistics

All state (velocities, moment estimates) is keyed by ``id(parameter)``, so
optimizers work with any layer type without needing to know its structure.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .tensor import Tensor, zeros

__all__ = [
    "Optimizer",
    "SGD",
    "Momentum",
    "RMSProp",
    "AdaGrad",
    "Adam",
    "get_optimizer",
    "OPTIMIZERS",
]


class Optimizer:
    """Base class for parameter update rules.

    ``build(parameters)`` allocates per-parameter state.  ``step(parameters,
    gradients)`` applies one update.  Parameters are modified **in place**
    (``Tensor.data[i] = ...``) so that layers, optimizer state, and any saved
    references all keep pointing at the same tensors.
    """

    def __init__(self, learning_rate: float = 0.01):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        self.learning_rate = float(learning_rate)
        #: Set by learning-rate schedules; the effective LR is
        #: ``learning_rate * lr_scale``.
        self.lr_scale = 1.0
        self.iterations = 0
        self._built = False

    @property
    def current_lr(self) -> float:
        """The learning rate actually used this step, after any schedule."""
        return self.learning_rate * self.lr_scale

    def build(self, parameters: Sequence[Tensor]) -> None:
        """Allocate state.  Called once by ``model.compile``."""
        self._built = True

    def step(self, parameters: Sequence[Tensor], gradients: Sequence[Tensor]) -> None:
        """Apply one update to every parameter."""
        raise NotImplementedError

    def _check(self, parameters: Sequence[Tensor], gradients: Sequence[Tensor]) -> None:
        if len(parameters) != len(gradients):
            raise ValueError(
                f"got {len(parameters)} parameters but {len(gradients)} gradients — "
                f"a layer's parameters() and gradients() must line up"
            )

    def state_dict(self) -> Dict[str, Any]:
        """Optimizer hyperparameters and step count, for checkpointing."""
        return {
            "type": type(self).__name__,
            "learning_rate": self.learning_rate,
            "lr_scale": self.lr_scale,
            "iterations": self.iterations,
            "config": self.config(),
        }

    def config(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        self.lr_scale = float(state.get("lr_scale", 1.0))
        self.iterations = int(state.get("iterations", 0))

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.config().items())
        return f"{type(self).__name__}({args})"


class SGD(Optimizer):
    """Plain stochastic gradient descent.

    The update rule, in full::

        W <- W - lr * dL/dW

    That is the entire algorithm.  Each parameter moves a fixed fraction of
    its own gradient, downhill.

    The learning rate ``lr`` is the one hyperparameter that matters most:

    * too large — the step overshoots the valley and the loss oscillates or
      diverges to ``nan``;
    * too small — training is correct but takes forever.

    Its weakness is that it uses only the *current* gradient.  In a ravine
    (steep across, shallow along) it bounces between the walls while creeping
    along the floor.  :class:`Momentum` fixes exactly that.
    """

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def step(self, parameters: Sequence[Tensor], gradients: Sequence[Tensor]) -> None:
        self._check(parameters, gradients)
        lr = self.current_lr
        for p, g in zip(parameters, gradients):
            pd, gd = p.data, g.data
            for i in range(len(pd)):
                pd[i] -= lr * gd[i]      # W <- W - lr * dL/dW
        self.iterations += 1


class Momentum(Optimizer):
    """SGD with momentum (optionally Nesterov).

    Instead of stepping along the gradient, keep a running *velocity* and
    step along that::

        v <- beta * v  +  g            # accumulate
        W <- W - lr * v                # move

    The physical analogy is a ball rolling downhill: ``beta`` (typically 0.9)
    is how much of the previous velocity survives, so it behaves like inertia.

    Why it helps.  Gradient components that keep pointing the same way add up
    across steps — with ``beta = 0.9`` a consistent direction reaches roughly
    ``1/(1-beta) = 10x`` the speed of plain SGD.  Components that flip sign
    each step (the ravine walls) cancel out instead.  So momentum damps the
    oscillation and accelerates the useful direction at the same time.

    Nesterov's variant evaluates the gradient at the point the momentum is
    about to carry you to, rather than where you are, giving it a chance to
    correct before overshooting.  In the standard reformulation::

        W <- W - lr * (beta * v_new + g)
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9,
                 nesterov: bool = False):
        super().__init__(learning_rate)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self._velocity: Dict[int, List[float]] = {}

    def build(self, parameters: Sequence[Tensor]) -> None:
        # One velocity buffer per parameter tensor, starting at rest.
        self._velocity = {id(p): [0.0] * p.size for p in parameters}
        self._built = True

    def step(self, parameters: Sequence[Tensor], gradients: Sequence[Tensor]) -> None:
        self._check(parameters, gradients)
        lr, beta = self.current_lr, self.momentum
        for p, g in zip(parameters, gradients):
            v = self._velocity.get(id(p))
            if v is None:                        # a parameter added after build()
                v = [0.0] * p.size
                self._velocity[id(p)] = v
            pd, gd = p.data, g.data
            for i in range(len(pd)):
                v[i] = beta * v[i] + gd[i]       # v <- beta*v + g
                if self.nesterov:
                    pd[i] -= lr * (beta * v[i] + gd[i])
                else:
                    pd[i] -= lr * v[i]           # W <- W - lr*v
        self.iterations += 1

    def config(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate, "momentum": self.momentum,
                "nesterov": self.nesterov}


class AdaGrad(Optimizer):
    """Adaptive gradients: divide by the accumulated gradient magnitude.

    ::

        s <- s + g^2
        W <- W - lr * g / (sqrt(s) + eps)

    Parameters that have seen large gradients get small steps, and vice
    versa — useful for sparse features, where rare-but-informative inputs
    would otherwise barely move.

    The flaw: ``s`` only ever grows, so the effective learning rate decays
    monotonically to zero and learning eventually stops.  :class:`RMSProp`
    fixes it by forgetting old gradients.
    """

    def __init__(self, learning_rate: float = 0.01, eps: float = 1e-8):
        super().__init__(learning_rate)
        self.eps = float(eps)
        self._sum_sq: Dict[int, List[float]] = {}

    def build(self, parameters: Sequence[Tensor]) -> None:
        self._sum_sq = {id(p): [0.0] * p.size for p in parameters}
        self._built = True

    def step(self, parameters: Sequence[Tensor], gradients: Sequence[Tensor]) -> None:
        self._check(parameters, gradients)
        lr, eps = self.current_lr, self.eps
        for p, g in zip(parameters, gradients):
            s = self._sum_sq.setdefault(id(p), [0.0] * p.size)
            pd, gd = p.data, g.data
            for i in range(len(pd)):
                s[i] += gd[i] * gd[i]
                pd[i] -= lr * gd[i] / (math.sqrt(s[i]) + eps)
        self.iterations += 1

    def config(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate, "eps": self.eps}


class RMSProp(Optimizer):
    """AdaGrad with an exponentially-decaying average instead of a sum.

    ::

        s <- rho * s + (1 - rho) * g^2
        W <- W - lr * g / (sqrt(s) + eps)

    Because ``s`` is a moving average rather than a running total, it tracks
    the *recent* gradient scale and stops decaying to zero.  ``rho = 0.9``
    means roughly the last ten steps dominate.
    """

    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9, eps: float = 1e-8):
        super().__init__(learning_rate)
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        self.rho = float(rho)
        self.eps = float(eps)
        self._avg_sq: Dict[int, List[float]] = {}

    def build(self, parameters: Sequence[Tensor]) -> None:
        self._avg_sq = {id(p): [0.0] * p.size for p in parameters}
        self._built = True

    def step(self, parameters: Sequence[Tensor], gradients: Sequence[Tensor]) -> None:
        self._check(parameters, gradients)
        lr, rho, eps = self.current_lr, self.rho, self.eps
        for p, g in zip(parameters, gradients):
            s = self._avg_sq.setdefault(id(p), [0.0] * p.size)
            pd, gd = p.data, g.data
            for i in range(len(pd)):
                s[i] = rho * s[i] + (1.0 - rho) * gd[i] * gd[i]
                pd[i] -= lr * gd[i] / (math.sqrt(s[i]) + eps)
        self.iterations += 1

    def config(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate, "rho": self.rho, "eps": self.eps}


class Adam(Optimizer):
    """Adaptive Moment Estimation — momentum and RMSProp combined.

    Adam tracks two exponential moving averages per parameter: the mean of
    the gradient (like momentum) and its uncentred variance (like RMSProp).

    **Step 1 — the moment estimates.**  With ``t`` the step number::

        m_t = beta1 * m_{t-1} + (1 - beta1) * g        # 1st moment: mean
        v_t = beta2 * v_{t-1} + (1 - beta2) * g^2      # 2nd moment: variance

    **Step 2 — bias correction.**  Both start at zero, which biases them
    towards zero for the first steps.  Concretely, after one step with
    ``beta1 = 0.9``, ``m_1 = 0.1*g`` — a tenth of the true gradient, so the
    model would barely move early on when it should move most.

    Expanding the recursion shows ``E[m_t] = (1 - beta1^t) * E[g]``, so
    dividing by that factor removes the bias exactly::

        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)

    At ``t = 1`` this rescales ``0.1*g`` back to ``g``.  As ``t`` grows,
    ``beta^t -> 0`` and the correction fades to a no-op.

    **Step 3 — the update.**::

        W <- W - lr * m_hat / (sqrt(v_hat) + eps)

    Read the ratio as "gradient divided by its own typical magnitude": each
    parameter gets a step size scaled to its own history, so a single
    learning rate works across parameters with very different gradient
    scales.  This is why ``lr = 0.001`` is a reliable default for Adam where
    SGD would need per-problem tuning.

    ``eps = 1e-8`` prevents division by zero when a gradient is consistently
    zero.
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, amsgrad: bool = False):
        super().__init__(learning_rate)
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.amsgrad = bool(amsgrad)
        self._m: Dict[int, List[float]] = {}
        self._v: Dict[int, List[float]] = {}
        self._v_max: Dict[int, List[float]] = {}

    def build(self, parameters: Sequence[Tensor]) -> None:
        self._m = {id(p): [0.0] * p.size for p in parameters}
        self._v = {id(p): [0.0] * p.size for p in parameters}
        if self.amsgrad:
            self._v_max = {id(p): [0.0] * p.size for p in parameters}
        self._built = True

    def step(self, parameters: Sequence[Tensor], gradients: Sequence[Tensor]) -> None:
        self._check(parameters, gradients)
        self.iterations += 1
        t = self.iterations
        lr, b1, b2, eps = self.current_lr, self.beta1, self.beta2, self.eps

        # Bias-correction denominators depend only on t, so compute them once
        # per step rather than once per element.
        bias1 = 1.0 - b1 ** t
        bias2 = 1.0 - b2 ** t

        for p, g in zip(parameters, gradients):
            key = id(p)
            m = self._m.setdefault(key, [0.0] * p.size)
            v = self._v.setdefault(key, [0.0] * p.size)
            pd, gd = p.data, g.data
            vmax = self._v_max.setdefault(key, [0.0] * p.size) if self.amsgrad else None

            for i in range(len(pd)):
                grad = gd[i]
                # 1. moment estimates
                m[i] = b1 * m[i] + (1.0 - b1) * grad
                v[i] = b2 * v[i] + (1.0 - b2) * grad * grad
                # 2. bias correction
                m_hat = m[i] / bias1
                if vmax is not None:
                    # AMSGrad: use the largest v seen so far, which makes the
                    # effective step size non-increasing and fixes a known
                    # non-convergence case in the original Adam proof.
                    if v[i] > vmax[i]:
                        vmax[i] = v[i]
                    v_hat = vmax[i] / bias2
                else:
                    v_hat = v[i] / bias2
                # 3. the update
                pd[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)

    def config(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate, "beta1": self.beta1,
                "beta2": self.beta2, "eps": self.eps, "amsgrad": self.amsgrad}


OPTIMIZERS: Dict[str, type] = {
    "sgd": SGD,
    "momentum": Momentum,
    "sgd_momentum": Momentum,
    "nesterov": Momentum,
    "adagrad": AdaGrad,
    "rmsprop": RMSProp,
    "adam": Adam,
}


def get_optimizer(spec, **kwargs) -> Optimizer:
    """Resolve ``"adam"`` / a class / an instance to an :class:`Optimizer`."""
    if isinstance(spec, Optimizer):
        return spec
    if isinstance(spec, type) and issubclass(spec, Optimizer):
        return spec(**kwargs)
    if isinstance(spec, str):
        key = spec.lower().strip()
        if key not in OPTIMIZERS:
            raise ValueError(
                f"unknown optimizer {spec!r}; available: "
                f"{', '.join(sorted(set(OPTIMIZERS)))}"
            )
        if key == "nesterov":
            kwargs.setdefault("nesterov", True)
        return OPTIMIZERS[key](**kwargs)
    raise TypeError(f"cannot interpret {spec!r} as an optimizer")
