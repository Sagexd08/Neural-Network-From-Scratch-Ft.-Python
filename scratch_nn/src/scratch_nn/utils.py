"""
utils.py
========

Seeding, numerical-stability guards, and the terminal plotting used instead
of matplotlib.

Numerical stability is not an afterthought in a from-scratch library — it is
most of the difference between a network that trains and one that prints
``nan`` on epoch 3.  The three classic failure points are:

1. ``exp(x)`` overflows for x around 710.  Softmax and sigmoid both call it.
2. ``log(0)`` is ``-inf``.  Every cross-entropy loss calls it.
3. Gradients that grow multiplicatively through layers ("exploding
   gradients") until they are ``inf``, at which point every weight becomes
   ``nan`` on the next update.

The helpers here address each one directly.
"""

from __future__ import annotations

import math
import random
import sys
from typing import Dict, Iterable, List, Sequence

from .tensor import Tensor

__all__ = [
    "set_seed",
    "safe_text",
    "GLYPHS",
    "UNICODE_OK",
    "get_rng",
    "EPS",
    "safe_log",
    "safe_exp",
    "safe_div",
    "clamp",
    "is_finite",
    "check_finite",
    "gradient_norm",
    "parameter_norm",
    "clip_gradient",
    "clip_by_value",
    "format_float",
    "bar_chart",
    "ascii_plot",
    "progress_bar",
    "format_table",
    "format_duration",
]

# The smallest number we are willing to feed to log(), and the smallest
# denominator we are willing to divide by.  1e-12 is comfortably above the
# 2.2e-16 resolution of a float64 near 1.0, so adding it does not vanish, yet
# small enough not to bias a loss value in any way you could notice.
EPS: float = 1e-12

# A single module-level RNG so that `set_seed` makes the entire library
# reproducible: shuffling, weight init and dropout masks all draw from here.
_RNG = random.Random()


# ---------------------------------------------------------------------------
# terminal glyphs
# ---------------------------------------------------------------------------
# Windows consoles default to a legacy code page (cp1252) that cannot encode
# box-drawing or block characters, and printing them raises
# UnicodeEncodeError.  Rather than crash — or force every user to run
# `chcp 65001` — detect what the attached stream can actually encode once, at
# import time, and fall back to pure ASCII when it cannot.

def _supports_unicode() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or ""
    try:
        "█─│└".encode(encoding)
        return True
    except (LookupError, UnicodeEncodeError, TypeError):
        return False


UNICODE_OK: bool = _supports_unicode()

#: Glyphs used by the plotting helpers, chosen to match the terminal.
GLYPHS: Dict[str, str] = (
    {"full": "█", "light": "░", "h": "─", "v": "│",
     "corner": "└", "bar": "│"}
    if UNICODE_OK else
    {"full": "#", "light": ".", "h": "-", "v": "|", "corner": "+", "bar": "|"}
)


def safe_text(text: str) -> str:
    """Strip characters the terminal cannot render, replacing them with ASCII.

    Applied to anything this library prints, so output degrades gracefully on
    a legacy console instead of raising ``UnicodeEncodeError`` mid-report.
    """
    if UNICODE_OK:
        return text
    replacements = {
        "█": "#", "░": ".", "─": "-", "│": "|",
        "└": "+", "—": "--", "→": "->", "≥": ">=",
        "≤": "<=", "μ": "u", "σ": "sigma", "²": "^2",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "replace").decode("ascii")


def set_seed(seed: int) -> None:
    """Make every random operation in the library reproducible.

    Seeds both the library's own generator and the global :mod:`random` one,
    so code that reaches for ``random.random()`` directly is covered too.

    >>> set_seed(42)
    """
    _RNG.seed(seed)
    random.seed(seed)


def get_rng() -> random.Random:
    """The shared generator used by initializers, shuffling, and dropout."""
    return _RNG


# ---------------------------------------------------------------------------
# scalar-level numerical guards
# ---------------------------------------------------------------------------

def safe_log(x: float, eps: float = EPS) -> float:
    """``log(x)`` that cannot return ``-inf``.

    Cross-entropy computes ``-log(p)``.  When the network is confident and
    wrong, ``p`` underflows to exactly 0.0 and ``log(0)`` is ``-inf``, which
    poisons the loss and every gradient downstream.  Clamping the input to a
    floor of ``eps`` caps the loss at ``-log(eps) ~= 27.6`` instead, which is
    a large-but-finite penalty — exactly the behaviour you want.
    """
    return math.log(x if x > eps else eps)


def safe_exp(x: float) -> float:
    """``exp(x)`` that saturates instead of raising ``OverflowError``.

    ``math.exp(710)`` raises; ``math.exp(-746)`` underflows to 0.0 quietly.
    We return ``inf``/``0.0`` at those boundaries so callers see a value they
    can reason about rather than an exception mid-forward-pass.
    """
    if x > 709.78:
        return math.inf
    if x < -745.0:
        return 0.0
    return math.exp(x)


def safe_div(numerator: float, denominator: float, eps: float = EPS) -> float:
    """Division with the denominator pushed away from zero.

    The sign of the denominator is preserved, so ``safe_div(1, -0.0)`` gives a
    large negative number rather than a large positive one.
    """
    if abs(denominator) < eps:
        denominator = eps if denominator >= 0 else -eps
    return numerator / denominator


def clamp(x: float, low: float, high: float) -> float:
    """Constrain ``x`` to ``[low, high]``."""
    if x < low:
        return low
    if x > high:
        return high
    return x


def is_finite(value) -> bool:
    """True when a float, iterable of floats, or Tensor contains no NaN/inf.

    NaN is detected with ``v != v``, which is true only for NaN.
    """
    if isinstance(value, Tensor):
        return value.is_finite()
    if isinstance(value, (int, float)):
        return not (value != value or value == math.inf or value == -math.inf)
    for v in value:
        if not is_finite(v):
            return False
    return True


def check_finite(value, name: str = "value") -> None:
    """Raise a clear error if ``value`` has gone NaN or infinite.

    Called at the end of each training batch.  Catching divergence at the
    batch where it starts is far more useful than discovering it 200 epochs
    later when every weight is ``nan``.
    """
    if not is_finite(value):
        raise FloatingPointError(
            f"{name} contains NaN or infinity. Common causes: learning rate too "
            f"high (try 10x smaller), exploding gradients (enable clip_norm), "
            f"log(0) in a loss (check label values are in range), or inputs that "
            f"were never normalized."
        )


# ---------------------------------------------------------------------------
# gradient hygiene
# ---------------------------------------------------------------------------

def gradient_norm(gradients: Sequence[Tensor]) -> float:
    """The global L2 norm across every gradient tensor.

    Gradients from all layers are treated as one long concatenated vector::

        ||g|| = sqrt( sum over every element g_i of g_i^2 )

    This single number is the standard health signal for training: it should
    shrink as the network converges.  A sudden spike means instability.
    """
    total = 0.0
    for g in gradients:
        for v in g.data:
            total += v * v
    return math.sqrt(total)


def parameter_norm(parameters: Sequence[Tensor]) -> float:
    """The global L2 norm across every parameter tensor.

    Useful next to :func:`gradient_norm`: if parameters grow without bound
    while the loss stalls, the model is diverging rather than learning.
    """
    return gradient_norm(parameters)


def clip_gradient(gradients: Sequence[Tensor], max_norm: float) -> float:
    """Rescale gradients in place so their global norm is at most ``max_norm``.

    The rule::

        if ||g|| > max_norm:
            g <- g * (max_norm / ||g||)

    Note that this scales *every* gradient by the same factor, so the
    direction of the update is preserved exactly — only its length is capped.
    That is why norm clipping is preferred over clipping each element
    independently, which would distort the direction.

    Returns the norm *before* clipping, so callers can log it.
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")
    total = gradient_norm(gradients)
    if total > max_norm and total > 0.0:
        scale = max_norm / total
        for g in gradients:
            g.mul_(scale)
    return total


def clip_by_value(gradients: Sequence[Tensor], limit: float) -> None:
    """Clamp every gradient element into ``[-limit, limit]`` in place.

    Cruder than norm clipping (it distorts the update direction) but it is a
    hard guarantee against a single enormous element, so it is a useful
    last-resort safety net.
    """
    for g in gradients:
        d = g.data
        for i in range(len(d)):
            if d[i] > limit:
                d[i] = limit
            elif d[i] < -limit:
                d[i] = -limit


# ---------------------------------------------------------------------------
# terminal rendering (this project's substitute for matplotlib)
# ---------------------------------------------------------------------------

def format_float(x: float, width: int = 8, places: int = 4) -> str:
    """Fixed-width float formatting that degrades gracefully for NaN/inf."""
    if x != x:
        return "nan".rjust(width)
    if x == math.inf:
        return "inf".rjust(width)
    if x == -math.inf:
        return "-inf".rjust(width)
    if x != 0 and (abs(x) < 1e-4 or abs(x) >= 1e6):
        return f"{x:.{places - 1}e}".rjust(width)
    return f"{x:.{places}f}".rjust(width)


def format_duration(seconds: float) -> str:
    """Human-readable elapsed time."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def bar_chart(values: Sequence[float], labels: Sequence[str] | None = None,
              width: int = 40, fill: str | None = None) -> str:
    """A horizontal bar chart, e.g. for a loss curve sampled over epochs.

    Bars are scaled so the largest value fills ``width`` characters::

        Epoch 100   0.4210  |████████████████
        Epoch 200   0.1130  |████
    """
    if not values:
        return "(no data)"
    fill = fill or GLYPHS["full"]
    if labels is None:
        labels = [f"[{i}]" for i in range(len(values))]
    if len(labels) != len(values):
        raise ValueError(
            f"bar_chart got {len(labels)} labels for {len(values)} values"
        )

    finite = [v for v in values if is_finite(v)]
    peak = max((abs(v) for v in finite), default=0.0)
    label_w = max(len(str(l)) for l in labels)
    lines: List[str] = []
    for label, value in zip(labels, values):
        if not is_finite(value):
            bar = "(nan)"
        elif peak == 0.0:
            bar = ""
        else:
            bar = fill * max(1 if value > 0 else 0,
                             int(round(abs(value) / peak * width)))
        lines.append(f"{str(label):<{label_w}}  {format_float(value)}  |{bar}")
    return "\n".join(lines)


def ascii_plot(series: Sequence[float], height: int = 12, width: int = 60,
               title: str = "", y_label: str = "") -> str:
    """A line plot drawn with block characters on a character grid.

    The series is resampled to ``width`` columns and quantised to ``height``
    rows.  Axis bounds are printed so the shape is readable as real numbers,
    not just a silhouette.
    """
    points = [v for v in series if is_finite(v)]
    if len(points) < 2:
        return f"{title}\n(not enough finite data to plot)" if title else \
               "(not enough finite data to plot)"

    # Resample to exactly `width` columns by bucket-averaging.
    if len(points) > width:
        bucket = len(points) / width
        sampled = []
        for i in range(width):
            lo = int(i * bucket)
            hi = max(lo + 1, int((i + 1) * bucket))
            chunk = points[lo:hi]
            sampled.append(sum(chunk) / len(chunk))
        points = sampled

    lo, hi = min(points), max(points)
    span = hi - lo
    if span == 0:
        span = 1.0
        lo -= 0.5

    grid = [[" "] * len(points) for _ in range(height)]
    prev_row = None
    for col, value in enumerate(points):
        frac = (value - lo) / span               # 0 at the bottom, 1 at the top
        row = height - 1 - int(round(frac * (height - 1)))
        row = max(0, min(height - 1, row))
        grid[row][col] = GLYPHS["h"]
        # Draw a vertical connector so steep segments do not look disjoint.
        if prev_row is not None and abs(row - prev_row) > 1:
            step = 1 if row > prev_row else -1
            for r in range(prev_row + step, row, step):
                grid[r][col] = GLYPHS["v"]
        prev_row = row

    axis_w = 10
    lines: List[str] = []
    if title:
        lines.append(title)
    for r, row_chars in enumerate(grid):
        if r == 0:
            tick = format_float(hi, axis_w - 2)
        elif r == height - 1:
            tick = format_float(lo, axis_w - 2)
        elif r == height // 2:
            tick = format_float(lo + span / 2, axis_w - 2)
        else:
            tick = " " * (axis_w - 2)
        lines.append(f"{tick} {GLYPHS['bar']}{''.join(row_chars)}")
    lines.append(" " * (axis_w - 1) + GLYPHS["corner"] + GLYPHS["h"] * len(points))
    footer = " " * axis_w + f"0{' ' * max(0, len(points) - 12)}{len(series)} steps"
    lines.append(footer)
    if y_label:
        lines.append(" " * axis_w + y_label)
    return "\n".join(lines)


def progress_bar(current: int, total: int, width: int = 24,
                 fill: str | None = None, empty: str | None = None) -> str:
    """``[████████░░░░░░░░]  8/20`` — used for the per-batch training readout."""
    if total <= 0:
        return ""
    fill = fill or GLYPHS["full"]
    empty = empty or GLYPHS["light"]
    frac = clamp(current / total, 0.0, 1.0)
    filled = int(round(frac * width))
    return f"[{fill * filled}{empty * (width - filled)}] {current:>{len(str(total))}}/{total}"


def format_table(rows: Sequence[Sequence[object]], headers: Sequence[str] | None = None,
                 align_right: bool = True) -> str:
    """Render a list of rows as an aligned text table with box-drawing rules."""
    all_rows = [list(map(str, r)) for r in rows]
    if headers:
        all_rows = [list(headers)] + all_rows
    if not all_rows:
        return ""
    n_cols = max(len(r) for r in all_rows)
    for r in all_rows:
        r.extend([""] * (n_cols - len(r)))
    widths = [max(len(r[c]) for r in all_rows) for c in range(n_cols)]

    def render(row: Sequence[str]) -> str:
        cells = []
        for c, cell in enumerate(row):
            cells.append(cell.rjust(widths[c]) if align_right and c > 0
                         else cell.ljust(widths[c]))
        return "  ".join(cells)

    lines = []
    if headers:
        lines.append(render(all_rows[0]))
        lines.append("-" * (sum(widths) + 2 * (n_cols - 1)))
        body = all_rows[1:]
    else:
        body = all_rows
    lines.extend(render(r) for r in body)
    return "\n".join(lines)
