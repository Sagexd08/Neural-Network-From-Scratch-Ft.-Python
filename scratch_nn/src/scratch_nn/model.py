"""
model.py
========

``Sequential``: a stack of layers, plus the orchestration of forward and
backward passes over the whole stack.

The whole of backpropagation, in one paragraph
----------------------------------------------
The loss ``L`` depends on the last layer's output, which depends on the one
before it, and so on back to the input.  The chain rule says the derivative
of a composition is the product of the derivatives::

    dL/dW_1 = dL/dA_n * dA_n/dA_{n-1} * ... * dA_2/dA_1 * dA_1/dW_1

Computing that product left to right means each layer needs only one thing
from its successor: ``dL/d(my output)``.  So the algorithm is:

1. Forward: run the layers in order, each caching its input.
2. Compute ``dL/dA_n`` from the loss.
3. Backward: run the layers in *reverse* order.  Each takes ``dL/d(output)``,
   stores its own parameter gradients, and returns ``dL/d(input)`` — which is
   precisely ``dL/d(output)`` for the layer before it.

That reversed loop is :meth:`Sequential.backward`, and it is nine lines long.
Every layer's contribution to it is derived in :mod:`scratch_nn.layers`.

Following one number through the machine
----------------------------------------
For a single XOR sample ``x = [0, 1]`` with target ``y = [1]``::

    x                                     [0, 1]
      -> Dense1: z1 = x @ W1 + b1         (1,2)@(2,8) + (8,) -> (1,8)
      -> ReLU:   a1 = max(0, z1)                              -> (1,8)
      -> Dense2: z2 = a1 @ W2 + b2        (1,8)@(8,1) + (1,) -> (1,1)
      -> Sigmoid: p = 1/(1+e^-z2)                             -> (1,1)
      -> BCE:    L = -[y log p + (1-y) log(1-p)]              -> scalar

    dL/dp  = (p - y) / (p(1-p))     from BinaryCrossEntropy.backward
    dL/dz2 = dL/dp * p(1-p)         from Sigmoid.backward  = p - y
    dL/dW2 = a1^T @ dL/dz2          from Dense.backward step 2
    dL/db2 = column sums of dL/dz2  step 3
    dL/da1 = dL/dz2 @ W2^T          step 4  -- handed to ReLU
    dL/dz1 = dL/da1 * (z1 > 0)      from ReLU.backward
    dL/dW1 = x^T @ dL/dz1
    dL/db1 = column sums of dL/dz1
    W <- W - lr * dL/dW             from the optimizer
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .activations import Softmax
from .layers import Dense, Layer
from .losses import CategoricalCrossEntropy, Loss, get_loss
from .tensor import Tensor, TensorShapeError, stack_rows
from .utils import check_finite, format_table

__all__ = ["Sequential", "Model"]

ArrayLike = Union[Tensor, Sequence[Sequence[float]], Sequence[float]]


def as_tensor(data: ArrayLike, name: str = "input") -> Tensor:
    """Coerce user input into a 2-D ``(batch, features)`` tensor.

    Accepts a Tensor, a list of rows, or a flat list (treated as one sample).
    Being liberal here means ``model.predict([[0, 1]])`` just works.
    """
    if isinstance(data, Tensor):
        t = data
    else:
        rows = list(data)
        if not rows:
            raise TensorShapeError(f"{name} is empty")
        t = stack_rows(rows) if isinstance(rows[0], (list, tuple)) else Tensor([rows])
    if t.ndim == 1:
        t = t.reshape(1, t.shape[0])
    return t


class Sequential:
    """A linear stack of layers.

    >>> model = Sequential([
    ...     Dense(2, 8, activation="relu"),
    ...     Dense(8, 1, activation="sigmoid"),
    ... ])
    >>> model.compile(loss="binary_cross_entropy", optimizer=SGD(0.1))

    Training lives in :mod:`scratch_nn.training`; :meth:`fit` delegates there
    so this class stays focused on the forward/backward mechanics.
    """

    def __init__(self, layers: Optional[Sequence[Layer]] = None, name: str = "Sequential"):
        self.layers: List[Layer] = list(layers or [])
        self.name = name
        self.loss: Optional[Loss] = None
        self.optimizer = None
        self.metrics: List[str] = []
        self._compiled = False
        #: Set during compile: True when the final layer is a softmax Dense
        #: paired with categorical cross-entropy, enabling the fused
        #: ``dZ = p - y`` gradient.
        self._fused_softmax_ce = False
        self._validate_chain()

    # -- construction -------------------------------------------------------

    def add(self, layer: Layer) -> "Sequential":
        """Append a layer, checking it lines up with the previous one."""
        self.layers.append(layer)
        self._validate_chain()
        self._compiled = False
        return self

    def _validate_chain(self) -> None:
        """Catch mismatched layer widths at construction, not mid-training."""
        prev: Optional[Dense] = None
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                if prev is not None and prev.output_size != layer.input_size:
                    raise TensorShapeError(
                        f"layer {i} Dense(input_size={layer.input_size}) does not "
                        f"match the previous Dense output_size={prev.output_size}"
                    )
                prev = layer

    # -- compile ------------------------------------------------------------

    def compile(self, loss, optimizer, metrics: Optional[Sequence[str]] = None) -> "Sequential":
        """Attach a loss and an optimizer, and prepare optimizer state.

        >>> model.compile(loss="binary_cross_entropy",
        ...               optimizer=Adam(learning_rate=0.001),
        ...               metrics=["accuracy"])
        """
        from .optimizers import Optimizer, get_optimizer

        if not self.layers:
            raise ValueError("cannot compile a model with no layers")

        self.loss = get_loss(loss)
        self.optimizer = (optimizer if isinstance(optimizer, Optimizer)
                          else get_optimizer(optimizer))
        self.metrics = list(metrics or [])

        # Detect the softmax + categorical-cross-entropy pairing so backward
        # can use the fused `p - y` gradient (see losses.py).
        last = self.layers[-1]
        self._fused_softmax_ce = (
            isinstance(self.loss, CategoricalCrossEntropy)
            and isinstance(last, Dense)
            and isinstance(last.activation, Softmax)
        )

        # Optimizers keep per-parameter state (momentum buffers, Adam's m/v),
        # so they need to see the parameter list once up front.
        self.optimizer.build(self.parameters())
        self._compiled = True
        return self

    def _require_compiled(self) -> None:
        if not self._compiled or self.loss is None or self.optimizer is None:
            raise RuntimeError(
                "model is not compiled — call model.compile(loss=..., optimizer=...) first"
            )

    # -- forward / backward -------------------------------------------------

    def forward(self, x: ArrayLike, training: bool = False) -> Tensor:
        """Run the input through every layer in order.

        ``training`` switches dropout on and makes BatchNorm use batch (not
        running) statistics.  Getting this flag wrong is the classic source
        of "why is my validation loss weird" bugs, so it is explicit.
        """
        out = as_tensor(x)
        for i, layer in enumerate(self.layers):
            try:
                out = layer.forward(out, training)
            except TensorShapeError as exc:
                raise TensorShapeError(
                    f"in layer {i} ({layer.describe()}): {exc}"
                ) from None
        return out

    def backward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Propagate ``dL/dy_pred`` back through the stack.

        Gradient buffers are accumulated into, so :meth:`zero_grad` must have
        been called first (``train_step`` does this).
        """
        self._require_compiled()
        y_true_t = as_tensor(y_true, "targets")

        if self._fused_softmax_ce:
            # dL/dZ = (p - y)/N directly; the softmax derivative is folded in,
            # so the last Dense layer must skip its activation backward step.
            grad = self.loss.backward_fused(y_pred, y_true_t)  # type: ignore[attr-defined]
            last = self.layers[-1]
            grad = last.backward(grad, fused_activation_grad=True)  # type: ignore[call-arg]
            remaining = self.layers[:-1]
        else:
            grad = self.loss.backward(y_pred, y_true_t)  # type: ignore[union-attr]
            remaining = self.layers

        # The reversed loop: each layer converts dL/d(its output) into
        # dL/d(its input), which is the next iteration's dL/d(output).
        for layer in reversed(remaining):
            grad = layer.backward(grad)
        return grad

    def zero_grad(self) -> None:
        """Clear every gradient buffer.  Call before each backward pass."""
        for layer in self.layers:
            layer.zero_grad()

    # -- loss / prediction --------------------------------------------------

    def compute_loss(self, y_pred: Tensor, y_true: ArrayLike,
                     include_regularization: bool = True) -> float:
        """Data loss plus the L1/L2 penalties contributed by each layer."""
        self._require_compiled()
        total = self.loss.forward(y_pred, as_tensor(y_true, "targets"))  # type: ignore[union-attr]
        if include_regularization:
            total += self.regularization_loss()
        return total

    def regularization_loss(self) -> float:
        """Sum of every layer's weight penalty."""
        return sum(layer.regularization_loss() for layer in self.layers)

    def predict(self, x: ArrayLike) -> Tensor:
        """Raw network output with dropout/BatchNorm in inference mode."""
        return self.forward(x, training=False)

    def predict_classes(self, x: ArrayLike, threshold: float = 0.5) -> List[int]:
        """Hard class labels.

        Binary (one output column): ``1`` when the probability exceeds
        ``threshold``.  Multiclass: the ``argmax`` column index.
        """
        probs = self.predict(x)
        if probs.shape[1] == 1:
            return [1 if v >= threshold else 0 for v in probs.data]
        rows, cols = probs.shape
        out = []
        for r in range(rows):
            base = r * cols
            best_j, best_v = 0, probs.data[base]
            for j in range(1, cols):
                if probs.data[base + j] > best_v:
                    best_j, best_v = j, probs.data[base + j]
            out.append(best_j)
        return out

    def predict_proba(self, x: ArrayLike) -> List[List[float]]:
        """Predicted probabilities as plain nested lists.

        For a single-output binary model, the two-column ``[P(0), P(1)]``
        form is returned, which is what callers usually want to display.
        """
        probs = self.predict(x)
        if probs.shape[1] == 1:
            return [[1.0 - p, p] for p in probs.data]
        return probs.tolist()  # type: ignore[return-value]

    # -- parameters ---------------------------------------------------------

    def parameters(self) -> List[Tensor]:
        """Every learnable tensor, in layer order."""
        return [p for layer in self.layers for p in layer.parameters()]

    def gradients(self) -> List[Tensor]:
        """Gradient buffers, aligned with :meth:`parameters`."""
        return [g for layer in self.layers for g in layer.gradients()]

    def num_parameters(self) -> int:
        """Total scalar parameter count."""
        return sum(p.size for p in self.parameters())

    # -- one training step --------------------------------------------------

    def train_step(self, x_batch: Tensor, y_batch: Tensor,
                   clip_norm: Optional[float] = None) -> Tuple[float, Tensor]:
        """Forward, loss, backward, update — the complete cycle for one batch.

        Returns ``(loss, predictions)``.
        """
        from .utils import clip_gradient

        self._require_compiled()

        # 1. Forward pass (training mode: dropout active).
        y_pred = self.forward(x_batch, training=True)

        # 2. Loss.
        loss_value = self.compute_loss(y_pred, y_batch)

        # 3. Reset gradients, then backpropagate into the buffers.
        self.zero_grad()
        self.backward(y_pred, y_batch)

        # 4. Optionally cap the global gradient norm before updating.
        if clip_norm is not None:
            clip_gradient(self.gradients(), clip_norm)

        # 5. Let the optimizer apply the update rule to every parameter.
        self.optimizer.step(self.parameters(), self.gradients())  # type: ignore[union-attr]

        check_finite(loss_value, "training loss")
        return loss_value, y_pred

    def fit(self, X, y, **kwargs):
        """Train the model.  See :func:`scratch_nn.training.fit` for options.

        >>> history = model.fit(X, y, epochs=1000, batch_size=32,
        ...                     validation_data=(X_val, y_val))
        """
        from .training import fit as _fit
        return _fit(self, X, y, **kwargs)

    def evaluate(self, X, y, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Loss and metrics on a dataset, in inference mode."""
        from .training import evaluate as _evaluate
        return _evaluate(self, X, y, batch_size=batch_size)

    # -- persistence --------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Architecture only — enough to rebuild the layer stack."""
        return {
            "name": self.name,
            "layers": [
                {"type": type(layer).__name__, "config": layer.config()}
                for layer in self.layers
            ],
        }

    def state_dict(self) -> Dict[str, Any]:
        """All learnable state, as JSON-friendly nested lists."""
        return {"layers": [layer.state_dict() for layer in self.layers]}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore weights produced by :meth:`state_dict`."""
        saved = state["layers"]
        if len(saved) != len(self.layers):
            raise ValueError(
                f"checkpoint has {len(saved)} layers but this model has "
                f"{len(self.layers)}"
            )
        for layer, sub in zip(self.layers, saved):
            layer.load_state_dict(sub)

    def clone_parameters(self) -> List[Tensor]:
        """A deep copy of every parameter — used by early stopping."""
        return [p.copy() for p in self.parameters()]

    def set_parameters(self, values: Sequence[Tensor]) -> None:
        """Copy values back into the live parameters, in place.

        In place, so optimizer state that references these tensors stays
        valid after an early-stopping restore.
        """
        params = self.parameters()
        if len(values) != len(params):
            raise ValueError(
                f"expected {len(params)} parameter tensors, got {len(values)}"
            )
        for p, v in zip(params, values):
            if p.shape != v.shape:
                raise TensorShapeError(
                    f"parameter shape mismatch: {p.shape} vs {v.shape}"
                )
            p.data[:] = v.data

    def save(self, path: str, **kwargs) -> None:
        """Write architecture + weights to JSON.  See :mod:`scratch_nn.serialization`."""
        from .serialization import save_model
        save_model(self, path, **kwargs)

    @classmethod
    def load(cls, path: str, **kwargs) -> "Sequential":
        """Rebuild a model from a JSON checkpoint."""
        from .serialization import load_model
        return load_model(path, **kwargs)

    def save_pickle(self, path: str, **kwargs) -> None:
        """Write the same payload as :meth:`save`, but as a ``.pkl`` file.

        JSON is the recommended format; use this when ``.pkl`` is what the
        surrounding tooling expects.  See :mod:`scratch_nn.serialization` for
        the security trade-off.
        """
        from .serialization import save_pickle
        save_pickle(self, path, **kwargs)

    @classmethod
    def load_pickle(cls, path: str, **kwargs) -> "Sequential":
        """Rebuild a model from a ``.pkl`` checkpoint you trust."""
        from .serialization import load_pickle
        return load_pickle(path, **kwargs)

    # -- display ------------------------------------------------------------

    def summary(self) -> str:
        """A Keras-style table of layers, output widths and parameter counts."""
        rows = []
        for i, layer in enumerate(self.layers):
            n_params = sum(p.size for p in layer.parameters())
            out_dim = layer.output_size if isinstance(layer, Dense) else "-"
            rows.append([f"{i}", layer.describe().split(" [")[0], str(out_dim),
                         f"{n_params:,}"])
        table = format_table(rows, headers=["#", "Layer", "Output", "Params"],
                             align_right=False)
        total = self.num_parameters()
        head = f"{self.name} — {len(self.layers)} layers, {total:,} parameters"
        compiled = (f"loss={self.loss.name}, optimizer={type(self.optimizer).__name__}"
                    if self._compiled else "not compiled")
        return f"{head}\n{'=' * max(len(head), 46)}\n{table}\n{'-' * 46}\n{compiled}"

    def __repr__(self) -> str:
        return f"Sequential({len(self.layers)} layers, {self.num_parameters()} params)"

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, i: int) -> Layer:
        return self.layers[i]


#: Alias, so ``from scratch_nn import Model`` reads naturally.
Model = Sequential
