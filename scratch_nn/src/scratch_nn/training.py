"""
training.py
===========

The training loop, plus early stopping, checkpointing and LR schedules.

One epoch is::

    shuffle the training data
    for each mini-batch:
        forward pass          -> predictions
        loss                  -> a scalar
        backward pass         -> gradients for every parameter
        optimizer step        -> updated parameters
    evaluate on train and validation
    run callbacks (early stopping, checkpointing, LR schedule)

Why shuffle every epoch: without it the batches are identical each time, so
the sequence of gradient estimates repeats exactly.  Re-shuffling makes the
noise independent between epochs, which both improves generalization and
prevents the model from locking into a cycle.

Why validate at all: training loss can only tell you how well the model has
memorised data it has already seen.  The gap between training and validation
loss is the entire signal for overfitting — when training loss keeps falling
while validation loss rises, the model has started memorising noise, and
:class:`EarlyStopping` exists to catch precisely that moment.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .metrics import METRIC_FUNCTIONS, accuracy
from .tensor import Tensor
from .utils import (ascii_plot, bar_chart, check_finite, format_duration,
                    format_float, get_rng, gradient_norm, progress_bar)

__all__ = [
    "History",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "step_decay",
    "exponential_decay",
    "cosine_decay",
    "fit",
    "evaluate",
]


class History:
    """Per-epoch metric records, and the plots made from them."""

    def __init__(self) -> None:
        self.records: Dict[str, List[float]] = {}
        self.epochs = 0
        self.elapsed = 0.0
        self.stopped_early = False
        self.best_epoch: Optional[int] = None

    def append(self, name: str, value: float) -> None:
        self.records.setdefault(name, []).append(value)

    def __getitem__(self, name: str) -> List[float]:
        return self.records[name]

    def __contains__(self, name: str) -> bool:
        return name in self.records

    def get(self, name: str, default=None):
        return self.records.get(name, default)

    def keys(self):
        return self.records.keys()

    def last(self, name: str) -> Optional[float]:
        values = self.records.get(name)
        return values[-1] if values else None

    def best(self, name: str, mode: str = "min") -> Optional[float]:
        values = self.records.get(name)
        if not values:
            return None
        return min(values) if mode == "min" else max(values)

    def plot(self, keys: Optional[Sequence[str]] = None, height: int = 12,
             width: int = 60) -> str:
        """ASCII line plots of the recorded curves — this project's matplotlib."""
        keys = keys or [k for k in ("loss", "val_loss", "accuracy", "val_accuracy")
                        if k in self.records]
        blocks = []
        for key in keys:
            values = self.records.get(key)
            if values and len(values) > 1:
                blocks.append(ascii_plot(values, height=height, width=width,
                                         title=f"{key} over {len(values)} epochs"))
        return "\n\n".join(blocks) if blocks else "(no history to plot)"

    def bars(self, key: str = "loss", samples: int = 10) -> str:
        """A sampled bar chart of one curve, in the style asked for by the spec."""
        values = self.records.get(key)
        if not values:
            return f"(no data for {key!r})"
        n = len(values)
        step = max(1, n // samples)
        picked = list(range(0, n, step))
        if picked[-1] != n - 1:
            picked.append(n - 1)
        return bar_chart([values[i] for i in picked],
                         [f"Epoch {i + 1:>4}  {key}" for i in picked])

    def summary(self) -> str:
        from .utils import format_table
        rows = []
        for name, values in self.records.items():
            if not values:
                continue
            mode = "max" if "acc" in name or name.endswith("f1") else "min"
            rows.append([name, format_float(values[0]), format_float(values[-1]),
                         format_float(min(values) if mode == "min" else max(values))])
        head = f"Trained {self.epochs} epochs in {format_duration(self.elapsed)}"
        if self.stopped_early:
            head += f" (stopped early; best epoch {self.best_epoch})"
        return head + "\n" + format_table(
            rows, headers=["metric", "first", "final", "best"])

    def __repr__(self) -> str:
        return f"History(epochs={self.epochs}, metrics={list(self.records)})"


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------

class Callback:
    """Hook interface invoked at fixed points in the training loop."""

    def on_train_begin(self, model, history: History) -> None: ...
    def on_epoch_begin(self, epoch: int, model, history: History) -> None: ...

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model,
                     history: History) -> bool:
        """Return True to request that training stop."""
        return False

    def on_train_end(self, model, history: History) -> None: ...


class EarlyStopping(Callback):
    """Stop when the monitored metric stops improving, and restore the best weights.

    The problem it solves: past some epoch, the model stops learning the
    signal and starts memorising the training set's noise.  Training loss
    keeps falling — memorisation always helps there — while validation loss
    turns and rises.  That turning point is the best model you will get, and
    training past it makes things strictly worse.

    Parameters
    ----------
    patience
        Epochs to wait for an improvement before stopping.  Some patience is
        essential because validation loss is noisy; a single bad epoch is not
        evidence of overfitting.
    min_delta
        The smallest change that counts as an improvement.  Guards against
        "improvements" of 1e-9 resetting the patience counter forever.
    restore_best_weights
        Roll the parameters back to the best epoch when stopping.  Without
        this you keep the *worst* weights of the patience window, which
        defeats the purpose.

    >>> EarlyStopping(patience=20, min_delta=0.001)
    """

    def __init__(self, monitor: str = "val_loss", patience: int = 10,
                 min_delta: float = 0.0, mode: str = "auto",
                 restore_best_weights: bool = True, verbose: bool = True):
        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = abs(float(min_delta))
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        if mode == "auto":
            # Accuracy-like metrics improve upwards; losses improve downwards.
            mode = "max" if ("acc" in monitor or monitor.endswith(("f1", "r2"))) else "min"
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min', 'max' or 'auto', got {mode!r}")
        self.mode = mode

        self.best: float = math.inf if mode == "min" else -math.inf
        self.best_epoch = 0
        self.wait = 0
        self.best_weights: Optional[List[Tensor]] = None
        self.stopped_epoch: Optional[int] = None

    def _improved(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta

    def on_train_begin(self, model, history: History) -> None:
        self.best = math.inf if self.mode == "min" else -math.inf
        self.wait = 0
        self.best_epoch = 0
        self.best_weights = None
        self.stopped_epoch = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model,
                     history: History) -> bool:
        current = logs.get(self.monitor)
        if current is None:
            if epoch == 1 and self.verbose:
                print(f"  [early stopping] metric {self.monitor!r} is not available "
                      f"(have: {', '.join(sorted(logs))}); callback disabled")
            return False

        if self._improved(current):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.clone_parameters()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                history.stopped_early = True
                history.best_epoch = self.best_epoch
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch}: {self.monitor} has not "
                          f"improved on {self.best:.6f} (epoch {self.best_epoch}) "
                          f"for {self.patience} epochs")
                return True
        return False

    def on_train_end(self, model, history: History) -> None:
        if self.restore_best_weights and self.best_weights is not None:
            model.set_parameters(self.best_weights)
            history.best_epoch = self.best_epoch
            if self.verbose and self.stopped_epoch:
                print(f"Restored weights from epoch {self.best_epoch} "
                      f"({self.monitor}={self.best:.6f})")


class ModelCheckpoint(Callback):
    """Save the model to disk whenever the monitored metric improves.

    Insurance against a crash and against overfitting at once: the file on
    disk is always the best model seen, not the most recent one.
    """

    def __init__(self, filepath: str, monitor: str = "val_loss",
                 mode: str = "auto", save_best_only: bool = True,
                 verbose: bool = False):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        if mode == "auto":
            mode = "max" if ("acc" in monitor or monitor.endswith(("f1", "r2"))) else "min"
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.saved_epoch: Optional[int] = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model,
                     history: History) -> bool:
        if not self.save_best_only:
            model.save(self.filepath)
            self.saved_epoch = epoch
            return False

        current = logs.get(self.monitor)
        if current is None:
            return False
        better = current < self.best if self.mode == "min" else current > self.best
        if better:
            self.best = current
            self.saved_epoch = epoch
            model.save(self.filepath)
            if self.verbose:
                print(f"  [checkpoint] epoch {epoch}: {self.monitor}={current:.6f} "
                      f"-> saved {self.filepath}")
        return False


def step_decay(initial: float = 1.0, drop: float = 0.5, every: int = 50) -> Callable[[int], float]:
    """Multiply the LR by ``drop`` every ``every`` epochs.

    The classic schedule: take big steps to find the right basin, then
    progressively smaller ones to settle into its floor rather than bouncing
    around it.
    """
    def schedule(epoch: int) -> float:
        return initial * (drop ** (epoch // every))
    return schedule


def exponential_decay(initial: float = 1.0, rate: float = 0.01) -> Callable[[int], float]:
    """Smooth decay: ``lr_scale = initial * e^(-rate * epoch)``."""
    def schedule(epoch: int) -> float:
        return initial * math.exp(-rate * epoch)
    return schedule


def cosine_decay(initial: float = 1.0, total_epochs: int = 100,
                 min_scale: float = 0.0) -> Callable[[int], float]:
    """Cosine annealing from ``initial`` down to ``min_scale``.

    ::

        scale = min + (init - min) * 0.5 * (1 + cos(pi * epoch/total))

    The curve decays slowly at first (keeping the exploratory phase long),
    then quickly through the middle, then flattens again at the end for a
    gentle landing.  Widely used, and usually beats step decay in practice.
    """
    def schedule(epoch: int) -> float:
        t = min(1.0, epoch / max(1, total_epochs))
        return min_scale + (initial - min_scale) * 0.5 * (1.0 + math.cos(math.pi * t))
    return schedule


class LearningRateScheduler(Callback):
    """Set ``optimizer.lr_scale`` from a function of the epoch number.

    The optimizer's effective rate is ``learning_rate * lr_scale``, so a
    schedule never destroys the base rate you configured.
    """

    def __init__(self, schedule: Callable[[int], float], verbose: bool = False):
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int, model, history: History) -> None:
        scale = float(self.schedule(epoch - 1))
        if scale <= 0:
            raise ValueError(f"learning-rate schedule returned {scale}; must be > 0")
        model.optimizer.lr_scale = scale
        if self.verbose:
            print(f"  [lr] epoch {epoch}: lr = {model.optimizer.current_lr:.6g}")

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model,
                     history: History) -> bool:
        logs["lr"] = model.optimizer.current_lr
        return False


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def _compute_metrics(model, X, y, names: Sequence[str], prefix: str = "") -> Dict[str, float]:
    """Evaluate loss plus the requested metrics on a dataset, in inference mode."""
    from .model import as_tensor
    y_pred = model.forward(X, training=False)
    y_true = as_tensor(y, "targets")
    out = {f"{prefix}loss": model.compute_loss(y_pred, y_true)}
    for name in names:
        fn = METRIC_FUNCTIONS.get(name)
        if fn is None:
            raise ValueError(
                f"unknown metric {name!r}; available: "
                f"{', '.join(sorted(METRIC_FUNCTIONS))}"
            )
        out[f"{prefix}{name}"] = fn(y_pred, y_true)
    return out


def fit(model, X, y, epochs: int = 100, batch_size: Optional[int] = 32,
        validation_data: Optional[Tuple[Any, Any]] = None,
        validation_split: float = 0.0, shuffle: bool = True,
        metrics: Optional[Sequence[str]] = None,
        callbacks: Optional[Sequence[Callback]] = None,
        verbose: int = 1, log_every: int = 1,
        clip_norm: Optional[float] = None,
        early_stopping: Optional[EarlyStopping] = None,
        seed: Optional[int] = None) -> History:
    """Train ``model`` on ``(X, y)``.

    Parameters
    ----------
    epochs
        Number of passes over the training set.
    batch_size
        ``None`` or ``len(X)`` for full-batch; 1 for pure SGD; 32 by default.
    validation_data
        ``(X_val, y_val)``, held out from training and evaluated each epoch.
    validation_split
        Alternative to ``validation_data``: carve this fraction off the end of
        ``X`` before training.
    metrics
        Names from :data:`scratch_nn.metrics.METRIC_FUNCTIONS`, e.g.
        ``["accuracy"]``.
    clip_norm
        Cap the global gradient norm at this value each step.
    verbose
        0 silent, 1 per-epoch lines, 2 adds a per-batch progress bar.

    Returns
    -------
    History

    >>> history = model.fit(X_train, y_train, epochs=1000, batch_size=32,
    ...                     validation_data=(X_val, y_val))
    """
    from .data import batch_iterator, train_test_split

    model._require_compiled()
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if len(X) == 0:
        raise ValueError("cannot train on an empty dataset")
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")

    if seed is not None:
        from .utils import set_seed
        set_seed(seed)

    metric_names = list(metrics if metrics is not None else model.metrics)
    callback_list: List[Callback] = list(callbacks or [])
    if early_stopping is not None:
        callback_list.append(early_stopping)

    # Resolve validation data.
    X_train, y_train = list(X), list(y)
    X_val = y_val = None
    if validation_data is not None:
        X_val, y_val = validation_data
    elif validation_split > 0:
        if not 0 < validation_split < 1:
            raise ValueError(
                f"validation_split must be in (0, 1), got {validation_split}")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, shuffle=shuffle)

    n_train = len(X_train)
    effective_bs = n_train if batch_size is None else min(int(batch_size), n_train)
    n_batches = math.ceil(n_train / effective_bs)

    history = History()
    rng = get_rng()
    started = time.time()

    if verbose:
        mode = ("full-batch" if effective_bs == n_train
                else "stochastic" if effective_bs == 1 else "mini-batch")
        print(f"Training on {n_train} samples"
              + (f", validating on {len(X_val)}" if X_val is not None else "")
              + f"  |  {mode}, batch_size={effective_bs}, {n_batches} batches/epoch")
        print(f"loss={model.loss.name}, optimizer={model.optimizer!r}, "
              f"{model.num_parameters():,} parameters")
        print("-" * 72)

    for cb in callback_list:
        cb.on_train_begin(model, history)

    stop_requested = False
    epoch = 0
    for epoch in range(1, epochs + 1):
        for cb in callback_list:
            cb.on_epoch_begin(epoch, model, history)

        # ---- one pass over the training data ------------------------------
        epoch_loss = 0.0
        seen = 0
        last_grad_norm = 0.0
        for b, (xb, yb) in enumerate(batch_iterator(X_train, y_train,
                                                    batch_size=effective_bs,
                                                    shuffle=shuffle, rng=rng)):
            batch_loss, _ = model.train_step(xb, yb, clip_norm=clip_norm)
            # Weight each batch's loss by its size, so a short final batch
            # does not skew the epoch average.
            epoch_loss += batch_loss * xb.shape[0]
            seen += xb.shape[0]
            if verbose >= 2:
                last_grad_norm = gradient_norm(model.gradients())
                print(f"\r  epoch {epoch:>4} {progress_bar(b + 1, n_batches)} "
                      f"loss={batch_loss:.4f} |g|={last_grad_norm:.3e}",
                      end="", flush=True)
        if verbose >= 2:
            print()

        train_batch_loss = epoch_loss / max(1, seen)
        check_finite(train_batch_loss, f"training loss at epoch {epoch}")

        # ---- evaluate -----------------------------------------------------
        # Re-evaluated in inference mode so dropout is off and the number is
        # comparable with the validation figure.
        logs = _compute_metrics(model, X_train, y_train, metric_names)
        logs["batch_loss"] = train_batch_loss
        if X_val is not None:
            logs.update(_compute_metrics(model, X_val, y_val, metric_names, prefix="val_"))

        for name, value in logs.items():
            history.append(name, value)
        history.epochs = epoch

        # ---- log ----------------------------------------------------------
        if verbose and (epoch % log_every == 0 or epoch == 1 or epoch == epochs):
            print(_format_epoch_line(epoch, epochs, logs))

        # ---- callbacks ----------------------------------------------------
        for cb in callback_list:
            if cb.on_epoch_end(epoch, logs, model, history):
                stop_requested = True
        if stop_requested:
            break

    history.elapsed = time.time() - started
    for cb in callback_list:
        cb.on_train_end(model, history)

    if verbose:
        print("-" * 72)
        print(f"Finished {history.epochs} epochs in {format_duration(history.elapsed)}")
    return history


def _format_epoch_line(epoch: int, total: int, logs: Dict[str, float]) -> str:
    """Render the per-epoch log line.

    ::

        Epoch 001/1000  loss: 0.6931  accuracy: 0.5000  val_loss: 0.6902
    """
    width = len(str(total))
    parts = [f"Epoch {epoch:0{width}d}/{total}"]
    order = ["loss", "accuracy", "acc", "precision", "recall", "f1",
             "mae", "mse", "rmse", "r2"]
    keys: List[str] = []
    for key in order:
        if key in logs:
            keys.append(key)
    for key in order:
        vk = f"val_{key}"
        if vk in logs:
            keys.append(vk)
    if "lr" in logs:
        keys.append("lr")
    for key in keys:
        parts.append(f"{key}: {logs[key]:.4f}")
    return "  ".join(parts)


def evaluate(model, X, y, batch_size: Optional[int] = None,
             metrics: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Loss and metrics on a dataset, with dropout and BatchNorm in inference mode.

    ``batch_size`` only splits the work up for memory; the result is identical
    either way.
    """
    model._require_compiled()
    metric_names = list(metrics if metrics is not None else model.metrics)
    if batch_size is None:
        return _compute_metrics(model, X, y, metric_names)

    # Size-weighted average across chunks, so a short last chunk is not
    # over-counted.
    from .data import batch_iterator
    totals: Dict[str, float] = {}
    seen = 0
    for xb, yb in batch_iterator(X, y, batch_size=batch_size, shuffle=False):
        chunk = _compute_metrics(model, xb, yb, metric_names)
        n = xb.shape[0]
        for k, v in chunk.items():
            totals[k] = totals.get(k, 0.0) + v * n
        seen += n
    return {k: v / seen for k, v in totals.items()}
