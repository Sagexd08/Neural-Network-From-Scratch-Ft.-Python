"""
serialization.py
================

Saving and loading models as JSON.

Why JSON and not pickle
-----------------------
``pickle`` would be one line of code, but it stores an opaque byte stream that
executes arbitrary code on load and tells a reader nothing.  A JSON checkpoint
from this library can be opened in any text editor, and you can *see* the
weights::

    {
      "format": "scratch_nn",
      "architecture": {
        "layers": [
          {"type": "Dense",
           "config": {"input_size": 2, "output_size": 8, "activation": "relu"}}
        ]
      },
      "weights": {"layers": [{"W": [[0.31, -0.22, ...]], "b": [0.0, ...]}]}
    }

Since the whole point of this project is that nothing is hidden, that
transparency is worth the file size.

What gets stored
----------------
* **architecture** — layer types and their constructor arguments, so the model
  can be rebuilt without the original source.
* **weights** — every learnable tensor, plus BatchNorm's running statistics
  (which are *not* learnable but are needed for correct inference).
* **optimizer state** — Adam's moment estimates and step count, so training
  can resume without the warm-up transient of a fresh optimizer.
* **preprocessing** — a fitted scaler and label encoder, so inference applies
  the exact transformation training used.  Skipping this is a classic
  deployment bug: the model receives raw units it has never seen.
* **metadata** — free-form; ``fit`` stores final metrics here.

Float round-tripping
--------------------
``repr(float)`` in Python 3 is guaranteed to round-trip exactly, and
``json.dump`` uses it, so saved weights reload bit-for-bit identically.  This
is what makes the round-trip test in ``tests/test_training.py`` demand
*exactly* equal predictions rather than merely close ones.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional

from .tensor import Tensor

__all__ = [
    "save_model",
    "load_model",
    "model_to_dict",
    "model_from_dict",
    "save_json",
    "load_json",
    "FORMAT_VERSION",
]

FORMAT_VERSION = "1.0"


def _build_layer(spec: Dict[str, Any]):
    """Reconstruct one layer from ``{"type": ..., "config": {...}}``."""
    from .layers import BatchNorm, Dense, Dropout, Flatten, Residual, Activation_

    registry = {
        "Dense": Dense,
        "Dropout": Dropout,
        "Flatten": Flatten,
        "BatchNorm": BatchNorm,
        "Activation_": Activation_,
        "Activation": Activation_,
    }
    kind = spec["type"]
    config = dict(spec.get("config", {}))

    if kind == "Residual":
        return Residual([_build_layer(s) for s in config.get("layers", [])])

    if kind not in registry:
        raise ValueError(
            f"unknown layer type {kind!r} in checkpoint; known types: "
            f"{', '.join(sorted(registry))}, Residual"
        )

    # An activation carrying its own arguments (LeakyReLU's alpha, say) is
    # stored separately, so rebuild the instance rather than the bare name.
    act_cfg = config.pop("activation_config", None)
    if act_cfg and "activation" in config:
        from .activations import get_activation
        config["activation"] = get_activation(config["activation"], **act_cfg)

    return registry[kind](**config)


def model_to_dict(model, include_optimizer: bool = True,
                  scaler=None, encoder=None,
                  metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Serialise a model (and optionally its preprocessing) to a plain dict."""
    payload: Dict[str, Any] = {
        "format": "scratch_nn",
        "version": FORMAT_VERSION,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "architecture": model.get_config(),
        "weights": model.state_dict(),
        "num_parameters": model.num_parameters(),
    }

    if model.loss is not None:
        payload["loss"] = {"type": type(model.loss).__name__,
                           "name": model.loss.name,
                           "config": model.loss.config()}

    if include_optimizer and model.optimizer is not None:
        state = model.optimizer.state_dict()
        # Adam/Momentum keep per-parameter buffers keyed by id(); id() is not
        # stable across processes, so they are stored positionally instead.
        buffers = _extract_optimizer_buffers(model)
        if buffers:
            state["buffers"] = buffers
        payload["optimizer"] = state

    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if encoder is not None:
        payload["encoder"] = encoder.state_dict()
    if metadata:
        payload["metadata"] = metadata
    return payload


def _extract_optimizer_buffers(model) -> Dict[str, List[List[float]]]:
    """Pull per-parameter optimizer state into parameter order."""
    opt = model.optimizer
    params = model.parameters()
    out: Dict[str, List[List[float]]] = {}
    for attr, key in (("_velocity", "velocity"), ("_m", "m"), ("_v", "v"),
                      ("_avg_sq", "avg_sq"), ("_sum_sq", "sum_sq"),
                      ("_v_max", "v_max")):
        store = getattr(opt, attr, None)
        if store:
            out[key] = [list(store.get(id(p), [0.0] * p.size)) for p in params]
    return out


def _restore_optimizer_buffers(model, buffers: Dict[str, List[List[float]]]) -> None:
    """Put positional buffers back into the optimizer's id()-keyed dicts."""
    opt = model.optimizer
    params = model.parameters()
    for attr, key in (("_velocity", "velocity"), ("_m", "m"), ("_v", "v"),
                      ("_avg_sq", "avg_sq"), ("_sum_sq", "sum_sq"),
                      ("_v_max", "v_max")):
        values = buffers.get(key)
        if values is None or not hasattr(opt, attr):
            continue
        if len(values) != len(params):
            raise ValueError(
                f"optimizer buffer {key!r} has {len(values)} entries but the model "
                f"has {len(params)} parameter tensors"
            )
        setattr(opt, attr, {id(p): list(v) for p, v in zip(params, values)})


def model_from_dict(payload: Dict[str, Any], compile_model: bool = True):
    """Rebuild a model from :func:`model_to_dict` output.

    Returns ``(model, extras)`` where ``extras`` holds any saved scaler,
    encoder and metadata.
    """
    from .data import LabelEncoder, MinMaxScaler, StandardScaler
    from .losses import get_loss
    from .model import Sequential
    from .optimizers import get_optimizer

    if payload.get("format") != "scratch_nn":
        raise ValueError(
            "this file was not written by scratch_nn (missing "
            "'format': 'scratch_nn')"
        )
    version = payload.get("version", "0")
    if version.split(".")[0] != FORMAT_VERSION.split(".")[0]:
        raise ValueError(
            f"checkpoint format version {version} is incompatible with this "
            f"library's {FORMAT_VERSION}"
        )

    arch = payload["architecture"]
    layers = [_build_layer(spec) for spec in arch["layers"]]
    model = Sequential(layers, name=arch.get("name", "Sequential"))
    model.load_state_dict(payload["weights"])

    if compile_model and "loss" in payload and "optimizer" in payload:
        loss_spec = payload["loss"]
        loss = get_loss(loss_spec["name"], **loss_spec.get("config", {}))
        opt_state = payload["optimizer"]
        optimizer = get_optimizer(
            opt_state["type"].lower(), **opt_state.get("config", {}))
        optimizer.load_state_dict(opt_state)
        model.compile(loss=loss, optimizer=optimizer)
        if "buffers" in opt_state:
            _restore_optimizer_buffers(model, opt_state["buffers"])

    extras: Dict[str, Any] = {}
    if "scaler" in payload:
        state = payload["scaler"]
        cls = StandardScaler if state["type"] == "StandardScaler" else MinMaxScaler
        extras["scaler"] = cls.from_state(state)
    if "encoder" in payload:
        extras["encoder"] = LabelEncoder.from_state(payload["encoder"])
    if "metadata" in payload:
        extras["metadata"] = payload["metadata"]

    return model, extras


def save_model(model, path: str, include_optimizer: bool = True,
               scaler=None, encoder=None,
               metadata: Optional[Dict[str, Any]] = None,
               indent: Optional[int] = 2) -> str:
    """Write a model to ``path`` as JSON.

    >>> model.save("models/xor.json")

    Set ``indent=None`` for a compact file when size matters more than
    readability.
    """
    payload = model_to_dict(model, include_optimizer=include_optimizer,
                            scaler=scaler, encoder=encoder, metadata=metadata)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Write to a temporary file then rename, so an interrupted save cannot
    # leave a half-written checkpoint where a valid one used to be.
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, allow_nan=False)
        os.replace(tmp, path)
    except ValueError as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        # allow_nan=False turns a diverged model into a clear error rather
        # than an invalid JSON file full of NaN tokens.
        raise ValueError(
            f"cannot save: the model contains NaN or infinite values ({exc}). "
            f"Training likely diverged — lower the learning rate or enable "
            f"gradient clipping."
        ) from None
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def load_model(path: str, compile_model: bool = True, with_extras: bool = False):
    """Load a model saved by :func:`save_model`.

    >>> model = Sequential.load("models/xor.json")

    With ``with_extras=True`` returns ``(model, extras)`` including any saved
    scaler and label encoder.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from None
    model, extras = model_from_dict(payload, compile_model=compile_model)
    return (model, extras) if with_extras else model


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    """Write any JSON-serialisable object (used for training histories)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent)


def load_json(path: str) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
