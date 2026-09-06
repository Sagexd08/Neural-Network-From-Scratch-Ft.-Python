"""
metrics.py
==========

Evaluation measures, implemented directly from their definitions.

Loss is what the optimizer minimises; metrics are what a human judges the
model by.  They differ deliberately: accuracy is not differentiable (it is a
step function of the predictions), so it cannot be a training objective, but
it is far more interpretable than cross-entropy.

The confusion matrix underlies all the classification metrics::

                      predicted 0     predicted 1
    actual 0              TN              FP
    actual 1              FN              TP

    accuracy  = (TP + TN) / total          "how often is it right?"
    precision = TP / (TP + FP)             "when it says yes, is it?"
    recall    = TP / (TP + FN)             "of the real yeses, how many found?"
    F1        = 2PR / (P + R)              harmonic mean of the two

Why accuracy alone is not enough: on a dataset that is 99% class 0, always
predicting 0 scores 99% accuracy while being useless.  Its precision and
recall on class 1 are both 0, which exposes it immediately.

Why the *harmonic* mean in F1: it is dominated by the smaller value, so
F1 is only high when precision and recall are *both* high.  The arithmetic
mean of precision 1.0 and recall 0.0 is a flattering 0.5; the harmonic mean
is 0.0, which is the honest answer.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .tensor import Tensor
from .utils import clamp, format_table, safe_div

__all__ = [
    "to_labels",
    "accuracy",
    "confusion_matrix",
    "precision",
    "recall",
    "f1_score",
    "classification_report",
    "format_confusion_matrix",
    "confusion_matrix_svg",
    "save_confusion_matrix_svg",
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "r_squared",
    "regression_report",
    "METRIC_FUNCTIONS",
]

Predictions = Union[Tensor, Sequence[Sequence[float]], Sequence[float]]


def to_labels(values: Predictions, threshold: float = 0.5) -> List[int]:
    """Convert probabilities, one-hot rows, or raw labels into integer classes.

    Handles the three shapes that show up in practice:

    * ``(n, 1)`` — sigmoid probabilities; thresholded.
    * ``(n, C)`` — softmax probabilities or one-hot targets; argmax.
    * ``(n,)``   — already labels; rounded to int.
    """
    if isinstance(values, Tensor):
        if values.ndim == 1:
            return [int(round(v)) for v in values.data]
        rows, cols = values.shape
        if cols == 1:
            return [1 if v >= threshold else 0 for v in values.data]
        out = []
        for r in range(rows):
            base = r * cols
            best_j, best_v = 0, values.data[base]
            for j in range(1, cols):
                if values.data[base + j] > best_v:
                    best_j, best_v = j, values.data[base + j]
            out.append(best_j)
        return out

    seq = list(values)
    if not seq:
        return []
    if isinstance(seq[0], (list, tuple)):
        out = []
        for row in seq:
            row = list(row)
            if len(row) == 1:
                out.append(1 if row[0] >= threshold else 0)
            else:
                out.append(max(range(len(row)), key=lambda j: row[j]))
        return out
    return [int(round(float(v))) for v in seq]


def _paired_labels(y_pred: Predictions, y_true: Predictions,
                   threshold: float) -> Tuple[List[int], List[int]]:
    pred = to_labels(y_pred, threshold)
    true = to_labels(y_true, threshold)
    if len(pred) != len(true):
        raise ValueError(
            f"got {len(pred)} predictions and {len(true)} targets — they must match"
        )
    if not pred:
        raise ValueError("cannot compute a metric over an empty set")
    return pred, true


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def accuracy(y_pred: Predictions, y_true: Predictions, threshold: float = 0.5) -> float:
    """Fraction of predictions that match the target exactly."""
    pred, true = _paired_labels(y_pred, y_true, threshold)
    correct = sum(1 for p, t in zip(pred, true) if p == t)
    return correct / len(pred)


def confusion_matrix(y_pred: Predictions, y_true: Predictions,
                     num_classes: Optional[int] = None,
                     threshold: float = 0.5) -> List[List[int]]:
    """``matrix[actual][predicted]`` counts.

    Rows are ground truth, columns are predictions, so the diagonal holds the
    correct answers and every off-diagonal cell names a specific mistake.
    """
    pred, true = _paired_labels(y_pred, y_true, threshold)
    if num_classes is None:
        num_classes = max(max(pred), max(true)) + 1
    if num_classes < 2:
        num_classes = 2
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for p, t in zip(pred, true):
        if not (0 <= t < num_classes and 0 <= p < num_classes):
            raise ValueError(
                f"label out of range: predicted {p}, actual {t}, but num_classes="
                f"{num_classes}"
            )
        matrix[t][p] += 1
    return matrix


def _per_class_counts(matrix: List[List[int]], cls: int) -> Tuple[int, int, int, int]:
    """Extract (TP, FP, FN, TN) for one class from a confusion matrix."""
    n = len(matrix)
    tp = matrix[cls][cls]
    fp = sum(matrix[r][cls] for r in range(n)) - tp     # predicted cls, wasn't
    fn = sum(matrix[cls][c] for c in range(n)) - tp     # was cls, predicted else
    tn = sum(matrix[r][c] for r in range(n) for c in range(n)) - tp - fp - fn
    return tp, fp, fn, tn


def precision(y_pred: Predictions, y_true: Predictions, positive_class: int = 1,
              average: Optional[str] = None, threshold: float = 0.5) -> float:
    """``TP / (TP + FP)`` — how trustworthy a positive prediction is.

    ``average="macro"`` averages the per-class scores with equal weight, which
    is the right choice for imbalanced multiclass problems.
    """
    matrix = confusion_matrix(y_pred, y_true, threshold=threshold)
    if average == "macro":
        return sum(
            safe_div(*_pr_pair(matrix, c)) for c in range(len(matrix))
        ) / len(matrix)
    tp, fp, _, _ = _per_class_counts(matrix, positive_class)
    return tp / (tp + fp) if (tp + fp) else 0.0


def _pr_pair(matrix: List[List[int]], cls: int) -> Tuple[float, float]:
    tp, fp, _, _ = _per_class_counts(matrix, cls)
    return (float(tp), float(tp + fp)) if (tp + fp) else (0.0, 1.0)


def recall(y_pred: Predictions, y_true: Predictions, positive_class: int = 1,
           average: Optional[str] = None, threshold: float = 0.5) -> float:
    """``TP / (TP + FN)`` — the share of real positives that were found."""
    matrix = confusion_matrix(y_pred, y_true, threshold=threshold)
    if average == "macro":
        total = 0.0
        for c in range(len(matrix)):
            tp, _, fn, _ = _per_class_counts(matrix, c)
            total += tp / (tp + fn) if (tp + fn) else 0.0
        return total / len(matrix)
    tp, _, fn, _ = _per_class_counts(matrix, positive_class)
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1_score(y_pred: Predictions, y_true: Predictions, positive_class: int = 1,
             average: Optional[str] = None, threshold: float = 0.5) -> float:
    """Harmonic mean of precision and recall: ``2PR / (P + R)``."""
    if average == "macro":
        matrix = confusion_matrix(y_pred, y_true, threshold=threshold)
        total = 0.0
        for c in range(len(matrix)):
            tp, fp, fn, _ = _per_class_counts(matrix, c)
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            total += 2 * p * r / (p + r) if (p + r) else 0.0
        return total / len(matrix)
    p = precision(y_pred, y_true, positive_class, threshold=threshold)
    r = recall(y_pred, y_true, positive_class, threshold=threshold)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def classification_report(y_pred: Predictions, y_true: Predictions,
                          class_names: Optional[Sequence[str]] = None,
                          threshold: float = 0.5) -> str:
    """A per-class precision/recall/F1 table plus overall accuracy."""
    matrix = confusion_matrix(y_pred, y_true, threshold=threshold)
    n_classes = len(matrix)
    names = list(class_names) if class_names else [f"class {i}" for i in range(n_classes)]

    rows = []
    total_support = 0
    macro_p = macro_r = macro_f = 0.0
    for c in range(n_classes):
        tp, fp, fn, _ = _per_class_counts(matrix, c)
        support = tp + fn
        total_support += support
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        macro_p += p
        macro_r += r
        macro_f += f
        rows.append([names[c] if c < len(names) else f"class {c}",
                     f"{p:.4f}", f"{r:.4f}", f"{f:.4f}", str(support)])

    correct = sum(matrix[i][i] for i in range(n_classes))
    rows.append(["", "", "", "", ""])
    rows.append(["macro avg", f"{macro_p / n_classes:.4f}",
                 f"{macro_r / n_classes:.4f}", f"{macro_f / n_classes:.4f}",
                 str(total_support)])
    rows.append(["accuracy", "", "", f"{correct / total_support:.4f}"
                 if total_support else "0.0000", str(total_support)])

    return format_table(rows, headers=["", "precision", "recall", "f1", "support"],
                        align_right=True)


def format_confusion_matrix(matrix: List[List[int]],
                            class_names: Optional[Sequence[str]] = None) -> str:
    """Render a confusion matrix as an aligned grid with a legend."""
    n = len(matrix)
    names = ([str(c) for c in class_names] if class_names
             else [str(i) for i in range(n)])
    cell_w = max(
        max((len(str(v)) for row in matrix for v in row), default=1),
        max(len(x) for x in names),
        3,
    )
    label_w = max(len("actual " + x) for x in names)

    lines = ["Confusion matrix  (rows = actual, columns = predicted)", ""]
    header = " " * label_w + "  " + "".join(f"{names[c]:>{cell_w + 2}}" for c in range(n))
    lines.append(header)
    for r in range(n):
        label = f"actual {names[r]}"
        cells = "".join(f"{matrix[r][c]:>{cell_w + 2}}" for c in range(n))
        lines.append(f"{label:<{label_w}}  {cells}")

    if n == 2:
        tn, fp = matrix[0][0], matrix[0][1]
        fn, tp = matrix[1][0], matrix[1][1]
        lines += ["", f"true negatives  {tn:>6}    false positives {fp:>6}",
                  f"false negatives {fn:>6}    true positives  {tp:>6}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the confusion graph
# ---------------------------------------------------------------------------
#
# The text grid above is exact but hard to *scan*: with more than a few classes
# the eye cannot tell 118 from 11 at a glance.  A heatmap fixes that by mapping
# count -> colour, so the structure of the errors is visible immediately: a
# strong diagonal means a good model, and a bright off-diagonal cell names the
# specific pair of classes being confused.
#
# We emit SVG rather than call a plotting library because this project has no
# dependencies.  SVG is just text, so a few hundred characters of string
# formatting produce a real vector image that any browser, editor, or README
# can display - and, like the JSON checkpoints, it stays inspectable.
#
# Colour scale
# ------------
# Cell intensity is normalised *per row* rather than globally.  Rows are the
# actual classes, so row-normalising asks "of the true class-i examples, where
# did they go?" - which is recall, the question a confusion matrix exists to
# answer.  Under global normalisation a large class saturates every colour and
# a rare class stays invisible however badly it is classified.
#
# Diagonal (correct) cells are shaded green and off-diagonal (error) cells red,
# so correctness reads as hue and magnitude as intensity.  Text flips to white
# on dark cells to stay legible.

_SVG_CORRECT = (16, 122, 74)     # green: the diagonal
_SVG_ERROR = (190, 52, 42)       # red:   everything off it


def _svg_escape(text: str) -> str:
    """Escape the five XML metacharacters so class names cannot break the SVG."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;"))


def _blend(rgb: Tuple[int, int, int], intensity: float) -> str:
    """Mix a colour toward white; ``intensity`` 0 -> white, 1 -> full colour."""
    t = clamp(intensity, 0.0, 1.0)
    r, g, b = (round(255 + (channel - 255) * t) for channel in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def confusion_matrix_svg(matrix: List[List[int]],
                         class_names: Optional[Sequence[str]] = None,
                         title: str = "Confusion matrix",
                         cell_size: int = 64,
                         normalize: bool = True) -> str:
    """Render a confusion matrix as a standalone SVG heatmap.

    Returns the SVG document as a string; :func:`save_confusion_matrix_svg`
    writes it to a file.

    >>> cm = confusion_matrix(predictions, y_test)
    >>> svg = confusion_matrix_svg(cm, class_names=["cat", "dog"])

    With ``normalize=True`` each cell also shows the row percentage, so a row
    reads as "of the true class-i examples, this is where they went".  Set it
    to ``False`` to show raw counts only.
    """
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("confusion matrix must be square and non-empty")

    names = ([str(c) for c in class_names] if class_names
             else [str(i) for i in range(n)])
    if len(names) < n:
        names = names + [str(i) for i in range(len(names), n)]

    # Layout: a label gutter on the left and top, the grid, then a legend.
    pad = 24
    label_w = max(90, 9 * max(len(x) for x in names[:n]) + 30)
    label_h = 54
    title_h = 40 if title else 8
    grid = n * cell_size
    # The title sits above the grid but is not constrained by it, so a long
    # title must be allowed to widen the canvas or it would be clipped.
    # 0.55em per character is a safe average for this font size.
    title_w = int(len(title) * 18 * 0.55) + pad * 2 if title else 0
    width = max(pad * 2 + label_w + grid, title_w)
    height = pad * 2 + title_h + label_h + grid + 54

    x0 = pad + label_w
    y0 = pad + title_h + label_h

    out: List[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif">'
    )
    out.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')

    if title:
        out.append(
            f'<text x="{pad}" y="{pad + 20}" font-size="18" font-weight="600" '
            f'fill="#1a1a1a">{_svg_escape(title)}</text>'
        )

    # Axis titles.
    out.append(
        f'<text x="{x0 + grid / 2:.1f}" y="{y0 - 32}" font-size="13" '
        f'font-weight="600" fill="#444" text-anchor="middle">predicted</text>'
    )
    out.append(
        f'<text x="{pad + 14}" y="{y0 + grid / 2:.1f}" font-size="13" '
        f'font-weight="600" fill="#444" text-anchor="middle" '
        f'transform="rotate(-90 {pad + 14} {y0 + grid / 2:.1f})">actual</text>'
    )

    # Column headers.
    for c in range(n):
        cx = x0 + c * cell_size + cell_size / 2
        out.append(
            f'<text x="{cx:.1f}" y="{y0 - 12}" font-size="12" fill="#333" '
            f'text-anchor="middle">{_svg_escape(names[c])}</text>'
        )

    # Row headers and cells.
    for r in range(n):
        row_total = sum(matrix[r])
        cy_label = y0 + r * cell_size + cell_size / 2 + 4
        out.append(
            f'<text x="{x0 - 12}" y="{cy_label:.1f}" font-size="12" fill="#333" '
            f'text-anchor="end">{_svg_escape(names[r])}</text>'
        )

        for c in range(n):
            count = matrix[r][c]
            # Row-normalised intensity: fraction of this true class.
            fraction = safe_div(count, row_total) if row_total else 0.0
            base = _SVG_CORRECT if r == c else _SVG_ERROR
            fill = _blend(base, fraction)
            x = x0 + c * cell_size
            y = y0 + r * cell_size
            out.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="{fill}" stroke="#ffffff" stroke-width="2"/>'
            )
            # Dark cells need light text to stay readable.
            text_fill = "#ffffff" if fraction > 0.55 else "#1a1a1a"
            cx = x + cell_size / 2
            if normalize and row_total:
                out.append(
                    f'<text x="{cx:.1f}" y="{y + cell_size / 2 - 2:.1f}" '
                    f'font-size="15" font-weight="600" fill="{text_fill}" '
                    f'text-anchor="middle">{count}</text>'
                )
                out.append(
                    f'<text x="{cx:.1f}" y="{y + cell_size / 2 + 15:.1f}" '
                    f'font-size="11" fill="{text_fill}" opacity="0.85" '
                    f'text-anchor="middle">{fraction * 100:.0f}%</text>'
                )
            else:
                out.append(
                    f'<text x="{cx:.1f}" y="{y + cell_size / 2 + 6:.1f}" '
                    f'font-size="15" font-weight="600" fill="{text_fill}" '
                    f'text-anchor="middle">{count}</text>'
                )

    # Legend: what the two hues mean, plus overall accuracy.
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(n))
    ly = y0 + grid + 30
    out.append(f'<rect x="{x0}" y="{ly - 11}" width="13" height="13" '
               f'fill="{_blend(_SVG_CORRECT, 0.85)}"/>')
    out.append(f'<text x="{x0 + 19}" y="{ly}" font-size="12" fill="#444">'
               f'correct</text>')
    out.append(f'<rect x="{x0 + 88}" y="{ly - 11}" width="13" height="13" '
               f'fill="{_blend(_SVG_ERROR, 0.85)}"/>')
    out.append(f'<text x="{x0 + 107}" y="{ly}" font-size="12" fill="#444">'
               f'misclassified</text>')
    out.append(
        f'<text x="{x0 + grid}" y="{ly}" font-size="12" fill="#444" '
        f'text-anchor="end">accuracy {safe_div(correct, total) * 100:.1f}% '
        f'({correct}/{total})</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def save_confusion_matrix_svg(matrix: List[List[int]], path: str,
                              class_names: Optional[Sequence[str]] = None,
                              **kwargs) -> str:
    """Write :func:`confusion_matrix_svg` output to ``path``; returns the path.

    >>> save_confusion_matrix_svg(cm, "reports/confusion.svg",
    ...                           class_names=["setosa", "versicolor"])
    """
    svg = confusion_matrix_svg(matrix, class_names=class_names, **kwargs)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)
    return path


# ---------------------------------------------------------------------------
# regression
# ---------------------------------------------------------------------------

def _flat_pairs(y_pred: Predictions, y_true: Predictions) -> Tuple[List[float], List[float]]:
    p = y_pred.data if isinstance(y_pred, Tensor) else _flatten(y_pred)
    t = y_true.data if isinstance(y_true, Tensor) else _flatten(y_true)
    if len(p) != len(t):
        raise ValueError(f"got {len(p)} predictions and {len(t)} targets")
    if not p:
        raise ValueError("cannot compute a metric over an empty set")
    return list(p), list(t)


def _flatten(values) -> List[float]:
    out: List[float] = []
    for v in values:
        if isinstance(v, (list, tuple)):
            out.extend(float(x) for x in v)
        else:
            out.append(float(v))
    return out


def mean_absolute_error(y_pred: Predictions, y_true: Predictions) -> float:
    """``mean |y_pred - y_true|`` — in the same units as the target."""
    p, t = _flat_pairs(y_pred, y_true)
    return math.fsum(abs(p[i] - t[i]) for i in range(len(p))) / len(p)


def mean_squared_error(y_pred: Predictions, y_true: Predictions) -> float:
    """``mean (y_pred - y_true)^2`` — punishes large errors disproportionately."""
    p, t = _flat_pairs(y_pred, y_true)
    return math.fsum((p[i] - t[i]) ** 2 for i in range(len(p))) / len(p)


def root_mean_squared_error(y_pred: Predictions, y_true: Predictions) -> float:
    """``sqrt(MSE)`` — MSE's outlier sensitivity, back in the target's units."""
    return math.sqrt(mean_squared_error(y_pred, y_true))


def r_squared(y_pred: Predictions, y_true: Predictions) -> float:
    """Coefficient of determination.

    ::

        R^2 = 1 - SS_res / SS_tot
        SS_res = sum (y - y_pred)^2       error the model leaves behind
        SS_tot = sum (y - mean(y))^2      error of always guessing the mean

    So R^2 is the fraction of the target's variance the model explains:
    1.0 is perfect, 0.0 is no better than predicting the mean, and negative
    means *worse* than predicting the mean (which is entirely possible).
    """
    p, t = _flat_pairs(y_pred, y_true)
    mean_t = math.fsum(t) / len(t)
    ss_res = math.fsum((t[i] - p[i]) ** 2 for i in range(len(t)))
    ss_tot = math.fsum((v - mean_t) ** 2 for v in t)
    if ss_tot == 0.0:
        # The target is constant, so there is no variance to explain.
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def regression_report(y_pred: Predictions, y_true: Predictions) -> str:
    """MAE / MSE / RMSE / R^2 in one table."""
    rows = [
        ["MAE", f"{mean_absolute_error(y_pred, y_true):.6f}"],
        ["MSE", f"{mean_squared_error(y_pred, y_true):.6f}"],
        ["RMSE", f"{root_mean_squared_error(y_pred, y_true):.6f}"],
        ["R^2", f"{r_squared(y_pred, y_true):.6f}"],
    ]
    return format_table(rows, headers=["metric", "value"])


#: Name -> function, for ``model.compile(metrics=[...])``.
METRIC_FUNCTIONS = {
    "accuracy": accuracy,
    "acc": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1_score,
    "f1_score": f1_score,
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error,
    "r2": r_squared,
    "r_squared": r_squared,
}
