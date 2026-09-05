"""
tensor.py
=========

A minimal N-dimensional tensor built on nested Python lists.

There is no NumPy here.  Every arithmetic operation below is a Python loop
over Python floats.  That is the whole point of this project: you should be
able to read any function in this file and see exactly which multiplications
and additions the machine performs.

Design
------
A ``Tensor`` stores its numbers in a *flat* list ``_data`` plus a ``shape``
tuple.  Nested lists are convenient to type::

    Tensor([[1, 2], [3, 4]])

but they are inconvenient to compute with, because indexing ``a[i][j]``
allocates intermediate list objects and makes strided access awkward.  So we
flatten on construction and reconstruct nested lists only when the user asks
for them (``tolist()``, ``__repr__``).

The mapping between an index tuple ``(i0, i1, ..., i_{n-1})`` and a flat
offset is the standard row-major ("C order") formula::

    offset = i0*s0 + i1*s1 + ... + i_{n-1}*s_{n-1}

where the *strides* are::

    s_{n-1} = 1
    s_k     = shape[k+1] * s_{k+1}

Concretely, for shape (2, 3)::

    strides = (3, 1)
    element (1, 2) lives at flat offset 1*3 + 2*1 = 5

Most neural-network work is 2-D (a batch of row vectors), so the 2-D paths
(``matmul``, ``transpose``) are written directly against the flat list for
speed rather than going through the general indexing machinery.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Iterator, List, Sequence, Tuple, Union

Number = Union[int, float]
NestedList = Union[Number, List[Any]]

__all__ = [
    "Tensor",
    "TensorShapeError",
    "zeros",
    "ones",
    "full",
    "eye",
    "rand",
    "randn",
    "arange",
    "from_flat",
    "stack_rows",
]


class TensorShapeError(ValueError):
    """Raised when an operation is impossible for the given shapes.

    A separate exception type (rather than bare ``ValueError``) lets tests
    assert that shape validation specifically is what failed.
    """


# ---------------------------------------------------------------------------
# helpers for converting nested lists <-> (flat data, shape)
# ---------------------------------------------------------------------------

def _infer_shape(data: NestedList) -> Tuple[int, ...]:
    """Return the shape of a nested list, validating that it is rectangular.

    ``[[1, 2], [3, 4]]``      -> ``(2, 2)``
    ``[1, 2, 3]``             -> ``(3,)``
    ``5``                     -> ``()``   (a scalar / 0-d tensor)
    ``[[1, 2], [3]]``         -> raises, because the rows are ragged
    """
    shape: List[int] = []
    node: Any = data
    while isinstance(node, (list, tuple)):
        shape.append(len(node))
        if len(node) == 0:
            # An empty dimension terminates the walk; e.g. [] -> (0,)
            break
        node = node[0]
    shape_t = tuple(shape)
    _validate_rectangular(data, shape_t, 0)
    return shape_t


def _validate_rectangular(node: Any, shape: Tuple[int, ...], depth: int) -> None:
    """Recursively confirm every sub-list matches the expected length."""
    if depth == len(shape):
        if isinstance(node, (list, tuple)):
            raise TensorShapeError(
                f"ragged nested list: found a sub-list at depth {depth} where a "
                f"number was expected (inferred shape {shape})"
            )
        if not isinstance(node, (int, float)) or isinstance(node, bool):
            raise TensorShapeError(
                f"tensor elements must be int or float, got {type(node).__name__!r} "
                f"({node!r}) at depth {depth}"
            )
        return
    if not isinstance(node, (list, tuple)):
        raise TensorShapeError(
            f"ragged nested list: expected a sequence of length {shape[depth]} at "
            f"depth {depth}, got scalar {node!r}"
        )
    if len(node) != shape[depth]:
        raise TensorShapeError(
            f"ragged nested list: expected length {shape[depth]} at depth {depth}, "
            f"got length {len(node)}"
        )
    for child in node:
        _validate_rectangular(child, shape, depth + 1)


def _flatten_into(node: Any, out: List[float]) -> None:
    """Append every leaf number of ``node`` to ``out`` in row-major order."""
    if isinstance(node, (list, tuple)):
        for child in node:
            _flatten_into(child, out)
    else:
        out.append(float(node))


def _strides_for(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Row-major strides: how far to step in the flat list per axis."""
    if not shape:
        return ()
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def _numel(shape: Tuple[int, ...]) -> int:
    """Number of elements implied by a shape.  ``()`` -> 1 (a scalar)."""
    total = 1
    for dim in shape:
        total *= dim
    return total


def _broadcast_shapes(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    """NumPy-style broadcasting rules, implemented by hand.

    Align shapes from the right.  Two dimensions are compatible when they are
    equal, or when one of them is 1 (that axis is then repeated).  Missing
    leading dimensions are treated as 1.

        (3, 4) with (4,)     -> (3, 4)
        (5, 1) with (1, 6)   -> (5, 6)
        (2, 3) with (4, 3)   -> error
    """
    result: List[int] = []
    for i in range(max(len(a), len(b))):
        dim_a = a[len(a) - 1 - i] if i < len(a) else 1
        dim_b = b[len(b) - 1 - i] if i < len(b) else 1
        if dim_a == dim_b:
            result.append(dim_a)
        elif dim_a == 1:
            result.append(dim_b)
        elif dim_b == 1:
            result.append(dim_a)
        else:
            raise TensorShapeError(
                f"cannot broadcast shapes {a} and {b}: dimension {i} counted from "
                f"the right has sizes {dim_a} and {dim_b}, and neither is 1"
            )
    result.reverse()
    return tuple(result)


def _unflatten(flat: Sequence[float], shape: Tuple[int, ...], offset: int = 0) -> NestedList:
    """Rebuild nested lists from flat data (the inverse of ``_flatten_into``)."""
    if not shape:
        return flat[offset]
    if len(shape) == 1:
        return list(flat[offset:offset + shape[0]])
    step = _numel(shape[1:])
    return [
        _unflatten(flat, shape[1:], offset + i * step)
        for i in range(shape[0])
    ]


# ---------------------------------------------------------------------------
# the Tensor class
# ---------------------------------------------------------------------------

class Tensor:
    """An immutable-by-convention N-dimensional array of Python floats.

    "Immutable by convention" means: operations return new tensors rather than
    mutating in place.  The two exceptions are :meth:`__setitem__` and the
    ``*_`` in-place helpers used by optimizers, which mutate deliberately so
    parameter updates do not allocate a fresh list every step.

    Examples
    --------
    >>> a = Tensor([[1, 2], [3, 4]])
    >>> b = Tensor([[5, 6], [7, 8]])
    >>> a.matmul(b).tolist()
    [[19.0, 22.0], [43.0, 50.0]]
    """

    __slots__ = ("_data", "_shape", "_strides")

    # -- construction -------------------------------------------------------

    def __init__(self, data: Union[NestedList, "Tensor"], shape: Tuple[int, ...] | None = None):
        if isinstance(data, Tensor):
            self._data = list(data._data)
            self._shape = data._shape
            self._strides = data._strides
            return

        if shape is not None:
            # Fast path used internally: caller supplies already-flat data.
            flat = [float(x) for x in data]  # type: ignore[union-attr]
            if len(flat) != _numel(shape):
                raise TensorShapeError(
                    f"flat data of length {len(flat)} does not fill shape {shape} "
                    f"(which needs {_numel(shape)} elements)"
                )
            self._data = flat
            self._shape = tuple(shape)
            self._strides = _strides_for(self._shape)
            return

        inferred = _infer_shape(data)
        flat_out: List[float] = []
        _flatten_into(data, flat_out)
        self._data = flat_out
        self._shape = inferred
        self._strides = _strides_for(inferred)

    # -- basic properties ---------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        """The size of each axis, e.g. ``(batch, features)``."""
        return self._shape

    @property
    def strides(self) -> Tuple[int, ...]:
        """Flat-offset step per axis (row-major)."""
        return self._strides

    @property
    def ndim(self) -> int:
        """Number of axes.  0 for a scalar, 1 for a vector, 2 for a matrix."""
        return len(self._shape)

    @property
    def size(self) -> int:
        """Total number of elements."""
        return len(self._data)

    @property
    def data(self) -> List[float]:
        """The underlying flat list.  Exposed for hot loops; treat as read-only."""
        return self._data

    def tolist(self) -> NestedList:
        """Return the contents as (possibly nested) plain Python lists."""
        return _unflatten(self._data, self._shape)

    def item(self) -> float:
        """Return the single element of a size-1 tensor as a Python float."""
        if len(self._data) != 1:
            raise TensorShapeError(
                f"item() requires exactly one element, this tensor has {len(self._data)} "
                f"(shape {self._shape})"
            )
        return self._data[0]

    def copy(self) -> "Tensor":
        """A deep copy: new flat list, same shape."""
        return Tensor(list(self._data), self._shape)

    # -- indexing -----------------------------------------------------------

    def _flat_index(self, idx: Tuple[int, ...]) -> int:
        """Translate an index tuple into a flat offset, with bounds checks."""
        if len(idx) != self.ndim:
            raise TensorShapeError(
                f"expected {self.ndim} indices for shape {self._shape}, got {len(idx)}"
            )
        offset = 0
        for axis, i in enumerate(idx):
            dim = self._shape[axis]
            if i < 0:
                i += dim  # negative indices count from the end, as in Python
            if not 0 <= i < dim:
                raise IndexError(
                    f"index {idx[axis]} is out of bounds for axis {axis} with size {dim}"
                )
            offset += i * self._strides[axis]
        return offset

    def __getitem__(self, idx: Union[int, Tuple[int, ...]]) -> Union[float, "Tensor"]:
        """``t[i]`` selects a sub-tensor; ``t[i, j]`` selects an element.

        A full index tuple returns a Python float.  A partial index returns a
        new ``Tensor`` view (a copy, since we do not implement real views).
        """
        if isinstance(idx, tuple):
            if len(idx) == self.ndim:
                return self._data[self._flat_index(idx)]
            if len(idx) > self.ndim:
                raise TensorShapeError(
                    f"too many indices: got {len(idx)} for shape {self._shape}"
                )
            # Partial index -> sub-tensor.
            sub_shape = self._shape[len(idx):]
            offset = 0
            for axis, i in enumerate(idx):
                dim = self._shape[axis]
                if i < 0:
                    i += dim
                if not 0 <= i < dim:
                    raise IndexError(
                        f"index {idx[axis]} is out of bounds for axis {axis} with size {dim}"
                    )
                offset += i * self._strides[axis]
            count = _numel(sub_shape)
            return Tensor(self._data[offset:offset + count], sub_shape)

        if self.ndim == 0:
            raise TensorShapeError("cannot index a 0-d tensor; use .item()")
        if self.ndim == 1:
            i = idx + self._shape[0] if idx < 0 else idx
            if not 0 <= i < self._shape[0]:
                raise IndexError(
                    f"index {idx} is out of bounds for axis 0 with size {self._shape[0]}"
                )
            return self._data[i]
        return self.__getitem__((idx,))

    def __setitem__(self, idx: Union[int, Tuple[int, ...]], value: Number) -> None:
        """In-place element assignment.  Only full indices are supported."""
        key = idx if isinstance(idx, tuple) else (idx,)
        self._data[self._flat_index(key)] = float(value)

    def __iter__(self) -> Iterator[Union[float, "Tensor"]]:
        """Iterate over the first axis, like NumPy and PyTorch do."""
        if self.ndim == 0:
            raise TensorShapeError("cannot iterate over a 0-d tensor")
        for i in range(self._shape[0]):
            yield self[i]

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TensorShapeError("a 0-d tensor has no len()")
        return self._shape[0]

    # -- shape manipulation -------------------------------------------------

    def reshape(self, *shape: Union[int, Sequence[int]]) -> "Tensor":
        """Reinterpret the same data under a new shape.

        The element count must match.  One dimension may be ``-1``, which is
        solved for::

            Tensor(range(12)).reshape(3, -1).shape  # -> (3, 4)
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            dims = list(shape[0])
        else:
            dims = list(shape)  # type: ignore[arg-type]

        if dims.count(-1) > 1:
            raise TensorShapeError(f"only one dimension may be -1, got {tuple(dims)}")
        if -1 in dims:
            known = 1
            for d in dims:
                if d != -1:
                    known *= d
            if known == 0 or self.size % known != 0:
                raise TensorShapeError(
                    f"cannot reshape {self.size} elements into {tuple(dims)}"
                )
            dims[dims.index(-1)] = self.size // known

        new_shape = tuple(int(d) for d in dims)
        if any(d < 0 for d in new_shape):
            raise TensorShapeError(f"negative dimension in reshape target {new_shape}")
        if _numel(new_shape) != self.size:
            raise TensorShapeError(
                f"cannot reshape tensor of {self.size} elements (shape {self._shape}) "
                f"into shape {new_shape} which needs {_numel(new_shape)}"
            )
        return Tensor(list(self._data), new_shape)

    def flatten(self) -> "Tensor":
        """Collapse to a 1-D tensor."""
        return Tensor(list(self._data), (self.size,))

    def transpose(self, *axes: int) -> "Tensor":
        """Permute axes.  With no arguments, reverse them (the usual matrix T).

        For the common 2-D case this is the loop::

            out[j, i] = in[i, j]
        """
        if self.ndim < 2:
            return self.copy()

        if not axes:
            order = tuple(range(self.ndim - 1, -1, -1))
        else:
            order = axes[0] if len(axes) == 1 and isinstance(axes[0], (list, tuple)) else axes  # type: ignore[assignment]
            order = tuple(int(a) for a in order)
            if sorted(order) != list(range(self.ndim)):
                raise TensorShapeError(
                    f"transpose axes {order} is not a permutation of "
                    f"{tuple(range(self.ndim))}"
                )

        # 2-D fast path: a straight double loop over the flat list.
        if self.ndim == 2 and order == (1, 0):
            rows, cols = self._shape
            src = self._data
            out = [0.0] * len(src)
            for i in range(rows):
                base = i * cols
                for j in range(cols):
                    out[j * rows + i] = src[base + j]
            return Tensor(out, (cols, rows))

        # General case: walk every output index and pull from the source.
        new_shape = tuple(self._shape[a] for a in order)
        new_strides = _strides_for(new_shape)
        out = [0.0] * self.size
        idx = [0] * self.ndim
        for flat in range(self.size):
            # Decode `flat` into an index of the *output* tensor.
            rem = flat
            for axis in range(self.ndim):
                idx[axis] = rem // new_strides[axis]
                rem -= idx[axis] * new_strides[axis]
            # Map back to the source offset through the permutation.
            src_off = 0
            for axis in range(self.ndim):
                src_off += idx[axis] * self._strides[order[axis]]
            out[flat] = self._data[src_off]
        return Tensor(out, new_shape)

    @property
    def T(self) -> "Tensor":
        """Shorthand for :meth:`transpose` with reversed axes."""
        return self.transpose()

    # -- element-wise machinery --------------------------------------------

    def map(self, fn: Callable[[float], float]) -> "Tensor":
        """Apply ``fn`` to every element, returning a new tensor.

        This is how every activation function is implemented.
        """
        return Tensor([fn(x) for x in self._data], self._shape)

    def map_(self, fn: Callable[[float], float]) -> "Tensor":
        """In-place :meth:`map`.  Returns ``self`` so calls can be chained."""
        data = self._data
        for i in range(len(data)):
            data[i] = fn(data[i])
        return self

    def _elementwise(self, other: Union["Tensor", Number], op: Callable[[float, float], float],
                     op_name: str) -> "Tensor":
        """Combine two tensors element-wise, broadcasting as needed."""
        if isinstance(other, (int, float)):
            value = float(other)
            return Tensor([op(x, value) for x in self._data], self._shape)

        if not isinstance(other, Tensor):
            raise TypeError(
                f"unsupported operand for {op_name}: Tensor and {type(other).__name__}"
            )

        # Identical shapes: the common case, and the fastest.
        if self._shape == other._shape:
            a, b = self._data, other._data
            return Tensor([op(a[i], b[i]) for i in range(len(a))], self._shape)

        return self._broadcast_elementwise(other, op, op_name)

    def _broadcast_elementwise(self, other: "Tensor", op: Callable[[float, float], float],
                               op_name: str) -> "Tensor":
        """Element-wise op where the shapes differ but are broadcast-compatible."""
        try:
            out_shape = _broadcast_shapes(self._shape, other._shape)
        except TensorShapeError as exc:
            raise TensorShapeError(f"{op_name}: {exc}") from None

        out_strides = _strides_for(out_shape)
        n_out = len(out_shape)

        # Effective strides: a broadcast axis (size 1 stretched to N) has
        # stride 0, so every output position re-reads the same source element.
        def effective(t: "Tensor") -> List[int]:
            pad = n_out - t.ndim
            eff = [0] * n_out
            for axis in range(t.ndim):
                eff[pad + axis] = 0 if t._shape[axis] == 1 else t._strides[axis]
            return eff

        eff_a = effective(self)
        eff_b = effective(other)
        a_data, b_data = self._data, other._data

        total = _numel(out_shape)
        out = [0.0] * total
        idx = [0] * n_out
        for flat in range(total):
            rem = flat
            off_a = off_b = 0
            for axis in range(n_out):
                i = rem // out_strides[axis]
                rem -= i * out_strides[axis]
                idx[axis] = i
                off_a += i * eff_a[axis]
                off_b += i * eff_b[axis]
            out[flat] = op(a_data[off_a], b_data[off_b])
        return Tensor(out, out_shape)

    # -- arithmetic operators ----------------------------------------------

    def add(self, other: Union["Tensor", Number]) -> "Tensor":
        """Element-wise addition (broadcasting)."""
        return self._elementwise(other, lambda x, y: x + y, "add")

    def sub(self, other: Union["Tensor", Number]) -> "Tensor":
        """Element-wise subtraction (broadcasting)."""
        return self._elementwise(other, lambda x, y: x - y, "sub")

    def mul(self, other: Union["Tensor", Number]) -> "Tensor":
        """Element-wise (Hadamard) product, or scalar multiplication."""
        return self._elementwise(other, lambda x, y: x * y, "mul")

    def div(self, other: Union["Tensor", Number]) -> "Tensor":
        """Element-wise division.  Division by zero raises rather than
        producing ``inf`` silently, because a silent ``inf`` in a gradient is
        far harder to debug than an exception at the source."""
        def safe_div(x: float, y: float) -> float:
            if y == 0.0:
                raise ZeroDivisionError(
                    "division by zero in Tensor.div; if this is expected, add an "
                    "epsilon to the denominator first"
                )
            return x / y
        return self._elementwise(other, safe_div, "div")

    def pow(self, exponent: Number) -> "Tensor":
        """Raise every element to ``exponent``."""
        e = float(exponent)
        return Tensor([x ** e for x in self._data], self._shape)

    def neg(self) -> "Tensor":
        """Element-wise negation."""
        return Tensor([-x for x in self._data], self._shape)

    def abs(self) -> "Tensor":
        """Element-wise absolute value."""
        return Tensor([abs(x) for x in self._data], self._shape)

    def sqrt(self) -> "Tensor":
        """Element-wise square root."""
        return Tensor([math.sqrt(x) for x in self._data], self._shape)

    def exp(self) -> "Tensor":
        """Element-wise ``e**x``, saturating instead of raising on overflow."""
        def _exp(x: float) -> float:
            if x > 709.0:      # math.exp overflows a bit above this
                return math.inf
            if x < -745.0:     # underflows to exactly 0.0 below this
                return 0.0
            return math.exp(x)
        return Tensor([_exp(x) for x in self._data], self._shape)

    def log(self, eps: float = 0.0) -> "Tensor":
        """Element-wise natural log.  ``eps`` guards ``log(0) = -inf``."""
        def _log(x: float) -> float:
            v = x + eps
            if v <= 0.0:
                raise ValueError(
                    f"log of non-positive value {x!r}; pass eps>0 to Tensor.log or "
                    f"clip the input first"
                )
            return math.log(v)
        return Tensor([_log(x) for x in self._data], self._shape)

    def clip(self, low: float, high: float) -> "Tensor":
        """Clamp every element into ``[low, high]``."""
        if low > high:
            raise ValueError(f"clip bounds are inverted: low={low} > high={high}")
        return Tensor([high if x > high else (low if x < low else x) for x in self._data],
                      self._shape)

    # Python operator sugar so the math in layers/losses reads like math.
    __add__ = add
    __sub__ = sub
    __mul__ = mul
    __truediv__ = div
    __pow__ = pow
    __neg__ = neg
    __abs__ = abs

    def __radd__(self, other: Number) -> "Tensor":
        return self.add(other)

    def __rsub__(self, other: Number) -> "Tensor":
        return Tensor([float(other) - x for x in self._data], self._shape)

    def __rmul__(self, other: Number) -> "Tensor":
        return self.mul(other)

    def __rtruediv__(self, other: Number) -> "Tensor":
        o = float(other)
        out = []
        for x in self._data:
            if x == 0.0:
                raise ZeroDivisionError("division by zero in scalar / Tensor")
            out.append(o / x)
        return Tensor(out, self._shape)

    # -- in-place variants used by optimizers -------------------------------

    def add_(self, other: Union["Tensor", Number]) -> "Tensor":
        """``self += other`` without allocating a new tensor.

        Optimizers call this once per parameter per step, so avoiding the
        allocation is worth the loss of immutability here.
        """
        data = self._data
        if isinstance(other, (int, float)):
            v = float(other)
            for i in range(len(data)):
                data[i] += v
        else:
            if other._shape != self._shape:
                raise TensorShapeError(
                    f"add_ requires identical shapes, got {self._shape} and {other._shape}"
                )
            od = other._data
            for i in range(len(data)):
                data[i] += od[i]
        return self

    def mul_(self, scalar: Number) -> "Tensor":
        """``self *= scalar`` in place."""
        v = float(scalar)
        data = self._data
        for i in range(len(data)):
            data[i] *= v
        return self

    def fill_(self, value: Number) -> "Tensor":
        """Set every element to ``value`` in place."""
        v = float(value)
        data = self._data
        for i in range(len(data)):
            data[i] = v
        return self

    def zero_(self) -> "Tensor":
        """Set every element to 0 in place (used to reset gradient buffers)."""
        return self.fill_(0.0)

    # -- reductions ---------------------------------------------------------

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Union[float, "Tensor"]:
        """Sum over one axis, or over everything when ``axis is None``.

        With ``axis=None`` this returns a plain float, which is what loss
        functions want.  With an axis it returns a Tensor, which is what bias
        gradients want (``db = sum over the batch axis``).
        """
        if axis is None:
            return math.fsum(self._data)  # fsum: less floating-point drift
        return self._reduce_axis(axis, keepdims, "sum")

    def mean(self, axis: int | None = None, keepdims: bool = False) -> Union[float, "Tensor"]:
        """Arithmetic mean over one axis, or over everything."""
        if self.size == 0:
            raise TensorShapeError("mean of an empty tensor is undefined")
        if axis is None:
            return math.fsum(self._data) / self.size
        summed = self._reduce_axis(axis, keepdims, "sum")
        n = self._shape[axis if axis >= 0 else axis + self.ndim]
        return summed.mul(1.0 / n)  # type: ignore[union-attr]

    def max(self, axis: int | None = None, keepdims: bool = False) -> Union[float, "Tensor"]:
        """Maximum over one axis, or over everything."""
        if self.size == 0:
            raise TensorShapeError("max of an empty tensor is undefined")
        if axis is None:
            return max(self._data)
        return self._reduce_axis(axis, keepdims, "max")

    def min(self, axis: int | None = None, keepdims: bool = False) -> Union[float, "Tensor"]:
        """Minimum over one axis, or over everything."""
        if self.size == 0:
            raise TensorShapeError("min of an empty tensor is undefined")
        if axis is None:
            return min(self._data)
        return self._reduce_axis(axis, keepdims, "min")

    def argmax(self, axis: int | None = None) -> Union[int, "Tensor"]:
        """Index of the maximum.  Flat index when ``axis is None``.

        For a 2-D tensor with ``axis=1`` this gives the predicted class of
        each row, which is exactly what multiclass accuracy needs.
        """
        if self.size == 0:
            raise TensorShapeError("argmax of an empty tensor is undefined")
        if axis is None:
            best_i, best_v = 0, self._data[0]
            for i in range(1, len(self._data)):
                if self._data[i] > best_v:
                    best_i, best_v = i, self._data[i]
            return best_i
        if self.ndim != 2 or axis not in (1, -1):
            raise TensorShapeError(
                "argmax with an axis is only implemented for 2-D tensors along axis=1"
            )
        rows, cols = self._shape
        out = []
        for r in range(rows):
            base = r * cols
            best_j, best_v = 0, self._data[base]
            for j in range(1, cols):
                v = self._data[base + j]
                if v > best_v:
                    best_j, best_v = j, v
            out.append(float(best_j))
        return Tensor(out, (rows,))

    def _reduce_axis(self, axis: int, keepdims: bool, how: str) -> "Tensor":
        """Shared implementation of sum/max/min along a single axis."""
        if axis < 0:
            axis += self.ndim
        if not 0 <= axis < self.ndim:
            raise TensorShapeError(
                f"axis {axis} is out of range for a {self.ndim}-d tensor with shape "
                f"{self._shape}"
            )
        if self._shape[axis] == 0:
            raise TensorShapeError(f"cannot reduce over empty axis {axis} of {self._shape}")

        out_shape = tuple(
            (1 if i == axis else d) if keepdims else d
            for i, d in enumerate(self._shape)
            if keepdims or i != axis
        )
        n_axis = self._shape[axis]
        # Outer = product of dims before `axis`; inner = product of dims after.
        outer = _numel(self._shape[:axis])
        inner = _numel(self._shape[axis + 1:])
        src = self._data
        out = [0.0] * (outer * inner)

        for o in range(outer):
            base_o = o * n_axis * inner
            for i in range(inner):
                start = base_o + i
                if how == "sum":
                    acc = 0.0
                    for k in range(n_axis):
                        acc += src[start + k * inner]
                elif how == "max":
                    acc = src[start]
                    for k in range(1, n_axis):
                        v = src[start + k * inner]
                        if v > acc:
                            acc = v
                else:  # min
                    acc = src[start]
                    for k in range(1, n_axis):
                        v = src[start + k * inner]
                        if v < acc:
                            acc = v
                out[o * inner + i] = acc
        return Tensor(out, out_shape)

    # -- linear algebra -----------------------------------------------------

    def matmul(self, other: "Tensor") -> "Tensor":
        """Matrix product.  ``(n, k) @ (k, m) -> (n, m)``.

        The triple loop below is the definition::

            C[i][j] = sum over k of A[i][k] * B[k][j]

        This is the single hottest function in the whole library — a forward
        pass through a Dense layer is one call to it — so it is written
        against the flat lists with the inner sum accumulated in a local
        variable and B accessed column-by-column via a stride.
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"matmul expects a Tensor, got {type(other).__name__}")
        if self.ndim != 2 or other.ndim != 2:
            raise TensorShapeError(
                f"matmul requires two 2-D tensors, got shapes {self._shape} and "
                f"{other._shape}; use reshape() first"
            )
        n, k = self._shape
        k2, m = other._shape
        if k != k2:
            raise TensorShapeError(
                f"matmul shape mismatch: {self._shape} @ {other._shape} — the inner "
                f"dimensions must agree ({k} != {k2})"
            )

        a, b = self._data, other._data
        out = [0.0] * (n * m)
        for i in range(n):
            row_off = i * k
            out_off = i * m
            for j in range(m):
                acc = 0.0
                b_off = j
                for p in range(k):
                    acc += a[row_off + p] * b[b_off]
                    b_off += m
                out[out_off + j] = acc
        return Tensor(out, (n, m))

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def dot(self, other: "Tensor") -> float:
        """Inner product of two 1-D tensors of equal length."""
        if self.ndim != 1 or other.ndim != 1:
            raise TensorShapeError(
                f"dot requires two 1-D tensors, got {self._shape} and {other._shape}"
            )
        if self._shape != other._shape:
            raise TensorShapeError(
                f"dot requires equal lengths, got {self._shape[0]} and {other._shape[0]}"
            )
        a, b = self._data, other._data
        return math.fsum(a[i] * b[i] for i in range(len(a)))

    def norm(self) -> float:
        """Euclidean (L2) norm of all elements, used for gradient clipping."""
        return math.sqrt(math.fsum(x * x for x in self._data))

    # -- predicates ---------------------------------------------------------

    def is_finite(self) -> bool:
        """True when no element is NaN or +/-inf."""
        for x in self._data:
            if x != x or x == math.inf or x == -math.inf:
                return False
        return True

    def equal(self, other: "Tensor", tol: float = 1e-9) -> bool:
        """Shape-and-value equality within ``tol``."""
        if not isinstance(other, Tensor) or self._shape != other._shape:
            return False
        a, b = self._data, other._data
        for i in range(len(a)):
            if abs(a[i] - b[i]) > tol:
                return False
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Tensor) and self.equal(other, tol=0.0)

    def __hash__(self):  # pragma: no cover - tensors are mutable, so unhashable
        raise TypeError("Tensor is unhashable (its contents can be mutated in place)")

    # -- display ------------------------------------------------------------

    def __repr__(self) -> str:
        if self.size <= 24:
            return f"Tensor({self._format(self.tolist())}, shape={self._shape})"
        return f"Tensor(<{self.size} elements>, shape={self._shape})"

    @staticmethod
    def _format(value: Any, places: int = 4) -> str:
        if isinstance(value, list):
            return "[" + ", ".join(Tensor._format(v, places) for v in value) + "]"
        return f"{value:.{places}g}"


# ---------------------------------------------------------------------------
# factory functions
# ---------------------------------------------------------------------------

def _as_shape(shape: Union[int, Sequence[int]], extra: Tuple[int, ...] = ()) -> Tuple[int, ...]:
    """Normalise ``zeros(2, 3)`` and ``zeros((2, 3))`` into ``(2, 3)``."""
    if isinstance(shape, int):
        return (shape,) + tuple(int(d) for d in extra)
    return tuple(int(d) for d in shape) + tuple(int(d) for d in extra)


def zeros(shape: Union[int, Sequence[int]], *extra: int) -> Tensor:
    """A tensor of the given shape filled with 0.0."""
    s = _as_shape(shape, extra)
    return Tensor([0.0] * _numel(s), s)


def ones(shape: Union[int, Sequence[int]], *extra: int) -> Tensor:
    """A tensor of the given shape filled with 1.0."""
    s = _as_shape(shape, extra)
    return Tensor([1.0] * _numel(s), s)


def full(shape: Union[int, Sequence[int]], value: Number) -> Tensor:
    """A tensor of the given shape filled with ``value``."""
    s = _as_shape(shape)
    return Tensor([float(value)] * _numel(s), s)


def eye(n: int) -> Tensor:
    """The ``n x n`` identity matrix."""
    data = [0.0] * (n * n)
    for i in range(n):
        data[i * n + i] = 1.0
    return Tensor(data, (n, n))


def rand(shape: Union[int, Sequence[int]], *extra: int, low: float = 0.0, high: float = 1.0,
         rng: random.Random | None = None) -> Tensor:
    """Uniform random values in ``[low, high)``."""
    s = _as_shape(shape, extra)
    r = rng or random
    return Tensor([r.uniform(low, high) for _ in range(_numel(s))], s)


def randn(shape: Union[int, Sequence[int]], *extra: int, mean: float = 0.0, std: float = 1.0,
          rng: random.Random | None = None) -> Tensor:
    """Gaussian random values with the given mean and standard deviation."""
    s = _as_shape(shape, extra)
    r = rng or random
    return Tensor([r.gauss(mean, std) for _ in range(_numel(s))], s)


def arange(start: Number, stop: Number | None = None, step: Number = 1) -> Tensor:
    """A 1-D tensor of evenly spaced values, like ``range`` but floats."""
    if stop is None:
        start, stop = 0, start
    values: List[float] = []
    x = float(start)
    if step == 0:
        raise ValueError("arange step must be non-zero")
    if step > 0:
        while x < stop:
            values.append(x)
            x += step
    else:
        while x > stop:
            values.append(x)
            x += step
    return Tensor(values, (len(values),))


def from_flat(data: Sequence[float], shape: Sequence[int]) -> Tensor:
    """Build a tensor directly from flat data plus a shape (no copying cost)."""
    return Tensor(list(data), tuple(int(d) for d in shape))


def stack_rows(rows: Iterable[Sequence[float]]) -> Tensor:
    """Stack equal-length sequences into a 2-D tensor.

    This is how a list of samples becomes a batch matrix.
    """
    rows = list(rows)
    if not rows:
        raise TensorShapeError("cannot stack zero rows")
    width = len(rows[0])
    flat: List[float] = []
    for i, row in enumerate(rows):
        if len(row) != width:
            raise TensorShapeError(
                f"row {i} has length {len(row)}, expected {width} (rows must be equal length)"
            )
        flat.extend(float(v) for v in row)
    return Tensor(flat, (len(rows), width))
