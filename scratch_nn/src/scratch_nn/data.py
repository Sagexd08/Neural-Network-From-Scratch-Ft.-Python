"""
data.py
=======

Loading, splitting, scaling, encoding and batching — without pandas.

The one rule that matters: **fit preprocessing on training data only.**

A scaler learns parameters (a mean, a standard deviation, a min and max).
If it learns them from the full dataset before splitting, information about
the validation and test sets leaks into the training pipeline, and your
measured performance is optimistically biased — the model has, indirectly,
already seen the data you are using to judge it.

The correct order is always:

1. split
2. ``scaler.fit(X_train)``
3. ``scaler.transform`` applied to train, validation *and* test

which is exactly what the ``fit``/``transform`` split of the API enforces.
:meth:`Scaler.transform` raises if called before ``fit``, so the mistake is
hard to make silently.
"""

from __future__ import annotations

import csv
import math
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from .tensor import Tensor, TensorShapeError, stack_rows
from .utils import get_rng

__all__ = [
    "Dataset",
    "load_csv",
    "save_csv",
    "train_test_split",
    "train_val_test_split",
    "shuffle_data",
    "batch_iterator",
    "StandardScaler",
    "MinMaxScaler",
    "LabelEncoder",
    "one_hot",
    "from_one_hot",
    "make_classification",
    "make_moons",
    "make_regression",
    "make_blobs",
    "xor_dataset",
]

Matrix = List[List[float]]


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def load_csv(path: str, target_column: Union[int, str, None] = -1,
             has_header: bool = True, delimiter: str = ",",
             skip_columns: Optional[Sequence[Union[int, str]]] = None,
             missing: str = "error") -> Tuple[Matrix, List[Any], List[str]]:
    """Read a CSV into ``(features, targets, feature_names)``.

    Parameters
    ----------
    target_column
        Index or header name of the label column.  ``-1`` means the last
        column; ``None`` means there is no label column (targets come back
        empty).
    missing
        What to do with unparseable or blank cells: ``"error"`` (default),
        ``"zero"``, or ``"mean"`` (column mean of the parseable values).

    Targets stay as raw strings when they are not numeric, so that
    :class:`LabelEncoder` can map them to integers afterwards.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, "r", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter=delimiter))
    rows = [r for r in rows if r and any(cell.strip() for cell in r)]
    if not rows:
        raise ValueError(f"{path} contains no data rows")

    if has_header:
        header = [h.strip() for h in rows[0]]
        body = rows[1:]
    else:
        header = [f"col{i}" for i in range(len(rows[0]))]
        body = rows
    if not body:
        raise ValueError(f"{path} has a header but no data rows")

    n_cols = len(header)
    for i, row in enumerate(body):
        if len(row) != n_cols:
            raise ValueError(
                f"{path} row {i + (2 if has_header else 1)} has {len(row)} fields, "
                f"expected {n_cols}"
            )

    def resolve(col: Union[int, str]) -> int:
        if isinstance(col, str):
            if col not in header:
                raise ValueError(f"column {col!r} not found; header is {header}")
            return header.index(col)
        return col if col >= 0 else n_cols + col

    target_idx = resolve(target_column) if target_column is not None else None
    skip = {resolve(c) for c in (skip_columns or ())}
    feature_idx = [i for i in range(n_cols) if i != target_idx and i not in skip]

    # First pass: parse, recording which cells could not be read as numbers.
    parsed: List[List[Optional[float]]] = []
    for row in body:
        parsed.append([_try_float(row[i]) for i in feature_idx])

    # Resolve missing values according to the policy.
    if missing == "mean":
        for c in range(len(feature_idx)):
            present = [r[c] for r in parsed if r[c] is not None]
            fill = sum(present) / len(present) if present else 0.0
            for r in parsed:
                if r[c] is None:
                    r[c] = fill
    elif missing == "zero":
        for r in parsed:
            for c in range(len(r)):
                if r[c] is None:
                    r[c] = 0.0
    else:
        for i, r in enumerate(parsed):
            for c, v in enumerate(r):
                if v is None:
                    raise ValueError(
                        f"{path} row {i + (2 if has_header else 1)}, column "
                        f"{header[feature_idx[c]]!r}: cannot parse "
                        f"{body[i][feature_idx[c]]!r} as a number. Pass "
                        f"missing='mean' or missing='zero' to impute instead."
                    )

    features: Matrix = [[float(v) for v in row] for row in parsed]  # type: ignore[arg-type]
    targets: List[Any] = []
    if target_idx is not None:
        for row in body:
            raw = row[target_idx].strip()
            num = _try_float(raw)
            targets.append(num if num is not None else raw)

    return features, targets, [header[i] for i in feature_idx]


def _try_float(cell: str) -> Optional[float]:
    text = cell.strip()
    if not text or text.lower() in {"na", "n/a", "nan", "null", "none", "?"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def save_csv(path: str, rows: Sequence[Sequence[Any]],
             header: Optional[Sequence[str]] = None) -> None:
    """Write rows to CSV, creating parent directories as needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# splitting, shuffling, batching
# ---------------------------------------------------------------------------

def shuffle_data(X: Sequence, y: Sequence,
                 rng: Optional[random.Random] = None) -> Tuple[List, List]:
    """Shuffle X and y together, keeping the pairing intact.

    Shuffles an index list rather than the data, which makes the paired
    permutation obviously correct.
    """
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    r = rng or get_rng()
    idx = list(range(len(X)))
    r.shuffle(idx)
    return [X[i] for i in idx], [y[i] for i in idx]


def train_test_split(X: Sequence, y: Sequence, test_size: float = 0.2,
                     shuffle: bool = True, stratify: bool = False,
                     rng: Optional[random.Random] = None) -> Tuple[List, List, List, List]:
    """Split into train and test sets.

    ``stratify=True`` preserves the class proportions in both halves, which
    matters for imbalanced data — a random split of a 95/5 dataset can easily
    leave the test set with no minority examples at all.

    Returns ``(X_train, X_test, y_train, y_test)``.
    """
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    n = len(X)
    if n < 2:
        raise ValueError(f"need at least 2 samples to split, got {n}")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size}")

    r = rng or get_rng()

    if stratify:
        groups: Dict[Any, List[int]] = {}
        for i, label in enumerate(y):
            groups.setdefault(_hashable(label), []).append(i)
        train_idx: List[int] = []
        test_idx: List[int] = []
        for indices in groups.values():
            if shuffle:
                indices = indices[:]
                r.shuffle(indices)
            cut = int(round(len(indices) * (1.0 - test_size)))
            cut = max(1, min(len(indices) - 1, cut)) if len(indices) > 1 else len(indices)
            train_idx.extend(indices[:cut])
            test_idx.extend(indices[cut:])
        if shuffle:
            r.shuffle(train_idx)
            r.shuffle(test_idx)
    else:
        idx = list(range(n))
        if shuffle:
            r.shuffle(idx)
        cut = int(round(n * (1.0 - test_size)))
        cut = max(1, min(n - 1, cut))
        train_idx, test_idx = idx[:cut], idx[cut:]

    return ([X[i] for i in train_idx], [X[i] for i in test_idx],
            [y[i] for i in train_idx], [y[i] for i in test_idx])


def _hashable(label: Any) -> Any:
    """Class labels arrive as scalars or one-hot rows; make both dict keys."""
    if isinstance(label, (list, tuple)):
        return tuple(label)
    return label


def train_val_test_split(X: Sequence, y: Sequence, val_size: float = 0.15,
                         test_size: float = 0.15, shuffle: bool = True,
                         stratify: bool = False,
                         rng: Optional[random.Random] = None) -> Tuple[List, ...]:
    """Three-way split.

    The three sets have distinct jobs, which is why you need all three:
    train fits the weights, validation guides decisions *about* training
    (early stopping, hyperparameters), and test is touched once at the very
    end.  Reusing validation as test overstates performance, because the
    stopping point was chosen to look good on exactly that data.

    Returns ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    if not 0.0 <= val_size < 1.0 or not 0.0 <= test_size < 1.0:
        raise ValueError("val_size and test_size must each be in [0, 1)")
    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size + test_size = {val_size + test_size} leaves no training data"
        )

    # Carve out the test set first, then split the remainder into train/val.
    if test_size > 0:
        X_rest, X_test, y_rest, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, stratify=stratify, rng=rng)
    else:
        X_rest, y_rest, X_test, y_test = list(X), list(y), [], []

    if val_size > 0:
        # Rescale: val_size is a fraction of the *original* set.
        val_fraction = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest, test_size=val_fraction, shuffle=shuffle,
            stratify=stratify, rng=rng)
    else:
        X_train, y_train, X_val, y_val = X_rest, y_rest, [], []

    return X_train, X_val, X_test, y_train, y_val, y_test


def batch_iterator(X: Sequence, y: Sequence, batch_size: Optional[int] = None,
                   shuffle: bool = True, rng: Optional[random.Random] = None,
                   drop_last: bool = False) -> Iterator[Tuple[Tensor, Tensor]]:
    """Yield ``(X_batch, y_batch)`` tensors.

    The batch size controls a real trade-off:

    * ``batch_size = len(X)`` — **full batch**.  Each step uses the exact
      gradient of the whole dataset: smooth, but one update per epoch.
    * ``batch_size = 1`` — **stochastic**.  Very noisy gradients, but many
      updates per epoch, and the noise itself helps escape sharp local minima.
    * ``batch_size = 32`` — **mini-batch**.  The usual compromise, and the
      default here.

    ``shuffle`` matters more than it looks: without it, every epoch sees the
    same batches in the same order, so the gradient noise is correlated across
    epochs and the model can lock into a cycle.
    """
    n = len(X)
    if n == 0:
        raise ValueError("cannot iterate over an empty dataset")
    if len(y) != n:
        raise ValueError(f"X has {n} rows but y has {len(y)}")

    size = n if batch_size is None else int(batch_size)
    if size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    size = min(size, n)

    idx = list(range(n))
    if shuffle:
        (rng or get_rng()).shuffle(idx)

    for start in range(0, n, size):
        chunk = idx[start:start + size]
        if drop_last and len(chunk) < size:
            break
        yield (_rows_to_tensor([X[i] for i in chunk]),
               _rows_to_tensor([y[i] for i in chunk]))


def _rows_to_tensor(rows: Sequence) -> Tensor:
    """Turn a list of samples into a 2-D tensor, accepting scalars or rows."""
    if isinstance(rows[0], (list, tuple)):
        return stack_rows(rows)
    return Tensor([[float(v)] for v in rows])


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------

class Scaler:
    """Base class enforcing the fit-then-transform discipline."""

    def __init__(self) -> None:
        self.fitted = False
        self.n_features = 0

    def fit(self, X: Sequence[Sequence[float]]) -> "Scaler":
        raise NotImplementedError

    def transform(self, X: Sequence[Sequence[float]]) -> Matrix:
        raise NotImplementedError

    def fit_transform(self, X: Sequence[Sequence[float]]) -> Matrix:
        """Fit on X then transform it.  **Only ever call this on training data.**"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Sequence[Sequence[float]]) -> Matrix:
        raise NotImplementedError

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(
                f"{type(self).__name__} has not been fitted. Call fit(X_train) "
                f"first — fitting on anything other than the training set leaks "
                f"information from your held-out data."
            )

    def _check_width(self, X: Sequence[Sequence[float]]) -> None:
        if X and len(X[0]) != self.n_features:
            raise TensorShapeError(
                f"{type(self).__name__} was fitted on {self.n_features} features "
                f"but received {len(X[0])}"
            )

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "Scaler":
        raise NotImplementedError


class StandardScaler(Scaler):
    """Standardization: rescale each feature to mean 0, standard deviation 1.

    ::

        x' = (x - mu) / sigma

    Why bother.  Gradient descent takes a step proportional to the gradient,
    and the gradient with respect to a weight is proportional to its input.
    A feature measured in the thousands therefore produces gradients a
    thousand times larger than one measured in units — so a learning rate
    that suits one will diverge or stall on the other.  Standardizing puts
    every feature on the same footing.

    A zero-variance feature (constant column) would divide by zero; its sigma
    is replaced by 1, which maps the whole column to 0 — a constant feature
    carries no information anyway.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean: List[float] = []
        self.std: List[float] = []

    def fit(self, X: Sequence[Sequence[float]]) -> "StandardScaler":
        if not X:
            raise ValueError("cannot fit a scaler on an empty dataset")
        n, f = len(X), len(X[0])
        self.n_features = f
        self.mean = [0.0] * f
        self.std = [1.0] * f
        for c in range(f):
            col = [float(row[c]) for row in X]
            mu = math.fsum(col) / n
            # Population variance (divide by n): we are describing this sample,
            # not estimating a wider population's parameter.
            var = math.fsum((v - mu) ** 2 for v in col) / n
            sigma = math.sqrt(var)
            self.mean[c] = mu
            self.std[c] = sigma if sigma > 1e-12 else 1.0
        self.fitted = True
        return self

    def transform(self, X: Sequence[Sequence[float]]) -> Matrix:
        self._check_fitted()
        self._check_width(X)
        return [[(float(row[c]) - self.mean[c]) / self.std[c]
                 for c in range(self.n_features)] for row in X]

    def inverse_transform(self, X: Sequence[Sequence[float]]) -> Matrix:
        """Map standardized values back to the original units."""
        self._check_fitted()
        return [[float(row[c]) * self.std[c] + self.mean[c]
                 for c in range(self.n_features)] for row in X]

    def state_dict(self) -> Dict[str, Any]:
        return {"type": "StandardScaler", "mean": self.mean, "std": self.std,
                "n_features": self.n_features}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "StandardScaler":
        s = cls()
        s.mean = list(state["mean"])
        s.std = list(state["std"])
        s.n_features = int(state["n_features"])
        s.fitted = True
        return s


class MinMaxScaler(Scaler):
    """Rescale each feature into a fixed range, by default ``[0, 1]``.

    ::

        x' = (x - min) / (max - min)          then mapped into [lo, hi]

    Compared with standardization: min-max guarantees bounded output (useful
    when a downstream component requires it) but is far more sensitive to
    outliers, since a single extreme value defines the range and squashes
    everything else into a sliver of it.  Standardization is the safer
    default; min-max is right when the bound genuinely matters.

    Values outside the fitted range at transform time are *not* clipped —
    they map outside ``[lo, hi]``, which is honest and lets you notice
    distribution shift.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        super().__init__()
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError(f"feature_range must be increasing, got {feature_range}")
        self.lo, self.hi = float(lo), float(hi)
        self.min: List[float] = []
        self.max: List[float] = []

    def fit(self, X: Sequence[Sequence[float]]) -> "MinMaxScaler":
        if not X:
            raise ValueError("cannot fit a scaler on an empty dataset")
        f = len(X[0])
        self.n_features = f
        self.min = [min(float(row[c]) for row in X) for c in range(f)]
        self.max = [max(float(row[c]) for row in X) for c in range(f)]
        self.fitted = True
        return self

    def transform(self, X: Sequence[Sequence[float]]) -> Matrix:
        self._check_fitted()
        self._check_width(X)
        span = self.hi - self.lo
        out: Matrix = []
        for row in X:
            scaled = []
            for c in range(self.n_features):
                width = self.max[c] - self.min[c]
                if width < 1e-12:
                    # Constant column: map to the middle of the range.
                    scaled.append((self.lo + self.hi) / 2.0)
                else:
                    unit = (float(row[c]) - self.min[c]) / width
                    scaled.append(self.lo + unit * span)
            out.append(scaled)
        return out

    def inverse_transform(self, X: Sequence[Sequence[float]]) -> Matrix:
        self._check_fitted()
        span = self.hi - self.lo
        out: Matrix = []
        for row in X:
            original = []
            for c in range(self.n_features):
                unit = (float(row[c]) - self.lo) / span
                original.append(unit * (self.max[c] - self.min[c]) + self.min[c])
            out.append(original)
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {"type": "MinMaxScaler", "min": self.min, "max": self.max,
                "lo": self.lo, "hi": self.hi, "n_features": self.n_features}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "MinMaxScaler":
        s = cls((state.get("lo", 0.0), state.get("hi", 1.0)))
        s.min = list(state["min"])
        s.max = list(state["max"])
        s.n_features = int(state["n_features"])
        s.fitted = True
        return s


# ---------------------------------------------------------------------------
# label encoding
# ---------------------------------------------------------------------------

class LabelEncoder:
    """Map arbitrary labels (``"setosa"``, ``"virginica"``) to ``0, 1, 2, ...``.

    Classes are sorted before numbering so the mapping is deterministic across
    runs — important for reproducibility and for reading saved models.
    """

    def __init__(self) -> None:
        self.classes: List[Any] = []
        self._to_index: Dict[Any, int] = {}
        self.fitted = False

    def fit(self, labels: Sequence[Any]) -> "LabelEncoder":
        unique = list({_hashable(l) for l in labels})
        try:
            unique.sort()
        except TypeError:
            unique.sort(key=str)   # mixed types: fall back to string order
        self.classes = unique
        self._to_index = {c: i for i, c in enumerate(unique)}
        self.fitted = True
        return self

    def transform(self, labels: Sequence[Any]) -> List[int]:
        if not self.fitted:
            raise RuntimeError("LabelEncoder.fit must be called before transform")
        out = []
        for l in labels:
            key = _hashable(l)
            if key not in self._to_index:
                raise ValueError(
                    f"unseen label {l!r}; the encoder knows {self.classes}"
                )
            out.append(self._to_index[key])
        return out

    def fit_transform(self, labels: Sequence[Any]) -> List[int]:
        return self.fit(labels).transform(labels)

    def inverse_transform(self, indices: Sequence[int]) -> List[Any]:
        """Turn class indices back into the original labels."""
        if not self.fitted:
            raise RuntimeError("LabelEncoder.fit must be called before inverse_transform")
        out = []
        for i in indices:
            i = int(i)
            if not 0 <= i < len(self.classes):
                raise ValueError(
                    f"class index {i} out of range for {len(self.classes)} classes"
                )
            out.append(self.classes[i])
        return out

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def state_dict(self) -> Dict[str, Any]:
        return {"classes": list(self.classes)}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "LabelEncoder":
        e = cls()
        e.classes = list(state["classes"])
        e._to_index = {c: i for i, c in enumerate(e.classes)}
        e.fitted = True
        return e


def one_hot(labels: Sequence[int], num_classes: Optional[int] = None) -> Matrix:
    """Turn class indices into one-hot rows: ``2`` with 4 classes -> ``[0,0,1,0]``.

    Why one-hot rather than feeding the integer directly: an integer label
    implies an ordering and a distance (class 3 is "closer" to class 2 than
    to class 0), which is false for categories.  One-hot puts every class at
    an equal distance from every other.  It is also exactly the target format
    that softmax + categorical cross-entropy expects.
    """
    labels = [int(l) for l in labels]
    if not labels:
        return []
    if num_classes is None:
        num_classes = max(labels) + 1
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    out: Matrix = []
    for l in labels:
        if not 0 <= l < num_classes:
            raise ValueError(
                f"label {l} is out of range for num_classes={num_classes}"
            )
        row = [0.0] * num_classes
        row[l] = 1.0
        out.append(row)
    return out


def from_one_hot(rows: Sequence[Sequence[float]]) -> List[int]:
    """Inverse of :func:`one_hot`: the index of the largest entry per row."""
    return [max(range(len(r)), key=lambda j: r[j]) for r in rows]


# ---------------------------------------------------------------------------
# synthetic datasets (so the examples need no downloads)
# ---------------------------------------------------------------------------

def xor_dataset() -> Tuple[Matrix, Matrix]:
    """The four points of the XOR truth table.

    XOR is the canonical proof that hidden layers matter: no straight line
    separates ``{(0,1), (1,0)}`` from ``{(0,0), (1,1)}``, so a single-layer
    perceptron provably cannot learn it.  It was this exact fact, published
    in 1969, that stalled neural-network research for over a decade.
    """
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [[0.0], [1.0], [1.0], [0.0]]
    return X, y


def make_classification(n_samples: int = 200, n_features: int = 2, n_classes: int = 2,
                        class_sep: float = 2.0, noise: float = 1.0,
                        rng: Optional[random.Random] = None) -> Tuple[Matrix, List[int]]:
    """Gaussian blobs, one per class, with a controllable separation.

    ``class_sep`` sets how far apart the cluster centres are, and ``noise``
    their spread — together they set how hard the problem is.
    """
    if n_samples < n_classes:
        raise ValueError(f"need at least {n_classes} samples for {n_classes} classes")
    r = rng or get_rng()
    centers = [[r.uniform(-class_sep, class_sep) for _ in range(n_features)]
               for _ in range(n_classes)]
    X: Matrix = []
    y: List[int] = []
    for i in range(n_samples):
        c = i % n_classes                      # exact class balance
        X.append([centers[c][f] + r.gauss(0.0, noise) for f in range(n_features)])
        y.append(c)
    return X, y


def make_moons(n_samples: int = 200, noise: float = 0.15,
               rng: Optional[random.Random] = None) -> Tuple[Matrix, List[int]]:
    """Two interleaving half-circles — not linearly separable.

    A step up from XOR: the decision boundary is curved, so it needs enough
    hidden capacity to bend around, and it makes an ASCII decision-boundary
    plot genuinely interesting.
    """
    r = rng or get_rng()
    X: Matrix = []
    y: List[int] = []
    half = n_samples // 2
    for i in range(half):
        angle = math.pi * i / max(1, half - 1)
        X.append([math.cos(angle) + r.gauss(0, noise),
                  math.sin(angle) + r.gauss(0, noise)])
        y.append(0)
    for i in range(n_samples - half):
        angle = math.pi * i / max(1, n_samples - half - 1)
        X.append([1.0 - math.cos(angle) + r.gauss(0, noise),
                  0.5 - math.sin(angle) + r.gauss(0, noise)])
        y.append(1)
    return X, y


def make_blobs(n_samples: int = 300, n_features: int = 2, centers: int = 3,
               cluster_std: float = 1.0, center_box: Tuple[float, float] = (-8.0, 8.0),
               rng: Optional[random.Random] = None) -> Tuple[Matrix, List[int]]:
    """Isotropic Gaussian clusters — the standard multiclass toy problem."""
    r = rng or get_rng()
    lo, hi = center_box
    centroids = [[r.uniform(lo, hi) for _ in range(n_features)] for _ in range(centers)]
    X: Matrix = []
    y: List[int] = []
    for i in range(n_samples):
        c = i % centers
        X.append([centroids[c][f] + r.gauss(0.0, cluster_std) for f in range(n_features)])
        y.append(c)
    return X, y


def make_regression(n_samples: int = 200, n_features: int = 1, noise: float = 0.1,
                    rng: Optional[random.Random] = None) -> Tuple[Matrix, Matrix]:
    """A noisy linear target, for exercising the regression path."""
    r = rng or get_rng()
    weights = [r.uniform(-3.0, 3.0) for _ in range(n_features)]
    bias = r.uniform(-1.0, 1.0)
    X: Matrix = []
    y: Matrix = []
    for _ in range(n_samples):
        row = [r.uniform(-3.0, 3.0) for _ in range(n_features)]
        target = bias + sum(row[f] * weights[f] for f in range(n_features))
        X.append(row)
        y.append([target + r.gauss(0.0, noise)])
    return X, y


class Dataset:
    """A convenience bundle of features, targets and their preprocessing state.

    Keeps the scaler and label encoder next to the data they were fitted on,
    so inference can reproduce the exact same transformations.

    >>> ds = Dataset(X, y).split(val_size=0.2, test_size=0.2)
    >>> ds.standardize()          # fits on train only, applies to all splits
    """

    def __init__(self, X: Sequence[Sequence[float]], y: Sequence,
                 feature_names: Optional[Sequence[str]] = None):
        self.X: Matrix = [[float(v) for v in row] for row in X]
        self.y: List[Any] = list(y)
        if len(self.X) != len(self.y):
            raise ValueError(f"X has {len(self.X)} rows but y has {len(self.y)}")
        self.feature_names = (list(feature_names) if feature_names
                              else [f"f{i}" for i in range(len(self.X[0]) if self.X else 0)])
        self.X_train: Matrix = []
        self.X_val: Matrix = []
        self.X_test: Matrix = []
        self.y_train: List[Any] = []
        self.y_val: List[Any] = []
        self.y_test: List[Any] = []
        self.scaler: Optional[Scaler] = None
        self.encoder: Optional[LabelEncoder] = None

    def split(self, val_size: float = 0.15, test_size: float = 0.15,
              stratify: bool = False, rng: Optional[random.Random] = None) -> "Dataset":
        """Partition into train / validation / test."""
        (self.X_train, self.X_val, self.X_test,
         self.y_train, self.y_val, self.y_test) = train_val_test_split(
            self.X, self.y, val_size=val_size, test_size=test_size,
            stratify=stratify, rng=rng)
        return self

    def _splits(self) -> List[Matrix]:
        return [s for s in (self.X_train, self.X_val, self.X_test) if s]

    def standardize(self) -> "Dataset":
        """Fit a :class:`StandardScaler` on the training split, apply to all."""
        self._require_split()
        self.scaler = StandardScaler().fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        if self.X_val:
            self.X_val = self.scaler.transform(self.X_val)
        if self.X_test:
            self.X_test = self.scaler.transform(self.X_test)
        return self

    def normalize(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> "Dataset":
        """Fit a :class:`MinMaxScaler` on the training split, apply to all."""
        self._require_split()
        self.scaler = MinMaxScaler(feature_range).fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        if self.X_val:
            self.X_val = self.scaler.transform(self.X_val)
        if self.X_test:
            self.X_test = self.scaler.transform(self.X_test)
        return self

    def encode_labels(self, one_hot_encode: bool = False) -> "Dataset":
        """Map labels to integers (optionally one-hot), fitting on all splits.

        Fitting the *encoder* on every split is safe and necessary — it learns
        only the set of class names, not any distributional information, and
        a class absent from training must still be representable.
        """
        self._require_split()
        self.encoder = LabelEncoder().fit(self.y_train + self.y_val + self.y_test)
        n = self.encoder.num_classes

        def convert(labels: List[Any]) -> List[Any]:
            if not labels:
                return []
            idx = self.encoder.transform(labels)  # type: ignore[union-attr]
            return one_hot(idx, n) if one_hot_encode else [[float(i)] for i in idx]

        self.y_train = convert(self.y_train)
        self.y_val = convert(self.y_val)
        self.y_test = convert(self.y_test)
        return self

    def _require_split(self) -> None:
        if not self.X_train:
            raise RuntimeError("call Dataset.split(...) before preprocessing")

    @property
    def num_features(self) -> int:
        return len(self.X[0]) if self.X else 0

    def summary(self) -> str:
        from .utils import format_table
        rows = [["train", str(len(self.X_train))],
                ["validation", str(len(self.X_val))],
                ["test", str(len(self.X_test))],
                ["features", str(self.num_features)]]
        if self.encoder:
            rows.append(["classes", str(self.encoder.num_classes)])
        if self.scaler:
            rows.append(["scaling", type(self.scaler).__name__])
        return format_table(rows, headers=["split", "count"])

    def __len__(self) -> int:
        return len(self.X)

    def __repr__(self) -> str:
        return (f"Dataset(n={len(self.X)}, features={self.num_features}, "
                f"train={len(self.X_train)}, val={len(self.X_val)}, "
                f"test={len(self.X_test)})")
