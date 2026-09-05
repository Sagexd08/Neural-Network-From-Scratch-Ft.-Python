"""
__main__.py
===========

Command-line interface.

    python -m scratch_nn xor
    python -m scratch_nn train --data data.csv --epochs 200
    python -m scratch_nn predict --model models/model.json --input "0,1"
    python -m scratch_nn gradient-check
    python -m scratch_nn test
    python -m scratch_nn demo
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from .utils import safe_text

_builtin_print = print


def print(*args, **kwargs):  # noqa: A001 - deliberate shadow of the builtin
    """``print`` that survives a legacy Windows code page.

    Every CLI message passes through :func:`~scratch_nn.utils.safe_text`, a
    no-op on a UTF-8 terminal that downgrades box-drawing and typographic
    characters to ASCII on a cp1252 one.  Without it the command dies with
    ``UnicodeEncodeError`` the moment it prints an em-dash.
    """
    _builtin_print(*[safe_text(a) if isinstance(a, str) else a for a in args],
                   **kwargs)


def _banner(title: str) -> str:
    return f"\n{'=' * 68}\n{title}\n{'=' * 68}"


# ---------------------------------------------------------------------------
# xor
# ---------------------------------------------------------------------------

def cmd_xor(args: argparse.Namespace) -> int:
    """Train a small network on XOR and print its truth table."""
    from .data import xor_dataset
    from .layers import Dense
    from .model import Sequential
    from .optimizers import get_optimizer
    from .utils import set_seed

    set_seed(args.seed)
    X, y = xor_dataset()

    print(_banner("XOR — the problem that needs a hidden layer"))
    print("\nXOR is not linearly separable: no straight line puts {(0,1),(1,0)}")
    print("on one side and {(0,0),(1,1)} on the other. A single-layer network")
    print("provably cannot learn it. Hidden layers can.\n")

    model = Sequential([
        Dense(2, args.hidden, activation="tanh", initialization="xavier"),
        Dense(args.hidden, args.hidden, activation="tanh", initialization="xavier"),
        Dense(args.hidden, 1, activation="sigmoid", initialization="xavier"),
    ])
    optimizer = get_optimizer(args.optimizer, learning_rate=args.lr)
    model.compile(loss="binary_cross_entropy", optimizer=optimizer,
                  metrics=["accuracy"])
    print(model.summary())
    print()

    history = model.fit(X, y, epochs=args.epochs, batch_size=4,
                        verbose=1, log_every=max(1, args.epochs // 10))

    print(_banner("Results"))
    predictions = model.predict(X)
    print("\n  a XOR b     probability   predicted   true")
    print("  " + "-" * 46)
    correct = 0
    for i, (row, target) in enumerate(zip(X, y)):
        p = predictions.data[i]
        label = 1 if p >= 0.5 else 0
        truth = int(target[0])
        correct += label == truth
        mark = "ok" if label == truth else "WRONG"
        print(f"  {int(row[0])} XOR {int(row[1])} -> {p:>10.4f}   "
              f"{label:^9}   {truth:^4}  {mark}")
    print(f"\n  accuracy: {correct}/4 = {correct / 4:.0%}")

    print("\n" + history.bars("loss", samples=10))
    if args.plot:
        print("\n" + history.plot(["loss"]))

    if args.save:
        model.save(args.save, metadata={"task": "xor", "accuracy": correct / 4})
        print(f"\nSaved to {args.save}")

    return 0 if correct == 4 else 1


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:
    """Train on a CSV file, end to end."""
    from .data import (Dataset, LabelEncoder, load_csv, make_classification,
                       one_hot)
    from .layers import Dense, Dropout
    from .metrics import classification_report, confusion_matrix, format_confusion_matrix
    from .model import Sequential
    from .optimizers import get_optimizer
    from .training import EarlyStopping
    from .utils import set_seed

    set_seed(args.seed)

    if args.data:
        print(f"Loading {args.data}")
        X, y_raw, feature_names = load_csv(
            args.data, target_column=args.target, missing=args.missing)
    else:
        print("No --data given; generating a synthetic classification dataset")
        X, y_raw = make_classification(n_samples=400, n_features=4, n_classes=3,
                                       class_sep=2.5, noise=1.0)
        feature_names = [f"f{i}" for i in range(4)]

    print(f"  {len(X)} samples, {len(X[0])} features")

    encoder = LabelEncoder().fit(y_raw)
    n_classes = encoder.num_classes
    print(f"  {n_classes} classes: {encoder.classes}")

    dataset = Dataset(X, y_raw, feature_names)
    dataset.split(val_size=args.val_size, test_size=args.test_size, stratify=True)
    dataset.standardize()          # fitted on the training split only
    dataset.encode_labels(one_hot_encode=n_classes > 2)
    print("\n" + dataset.summary())

    n_out = n_classes if n_classes > 2 else 1
    out_act = "softmax" if n_classes > 2 else "sigmoid"
    loss = "categorical_cross_entropy" if n_classes > 2 else "binary_cross_entropy"

    layers: List = [Dense(dataset.num_features, args.hidden, activation="relu",
                          initialization="he", l2=args.l2)]
    if args.dropout > 0:
        layers.append(Dropout(args.dropout))
    layers.append(Dense(args.hidden, args.hidden, activation="relu",
                        initialization="he", l2=args.l2))
    if args.dropout > 0:
        layers.append(Dropout(args.dropout))
    layers.append(Dense(args.hidden, n_out, activation=out_act,
                        initialization="xavier"))

    model = Sequential(layers)
    model.compile(loss=loss,
                  optimizer=get_optimizer(args.optimizer, learning_rate=args.lr),
                  metrics=["accuracy"])
    print("\n" + model.summary() + "\n")

    history = model.fit(
        dataset.X_train, dataset.y_train,
        epochs=args.epochs, batch_size=args.batch_size,
        validation_data=(dataset.X_val, dataset.y_val) if dataset.X_val else None,
        early_stopping=EarlyStopping(patience=args.patience, min_delta=1e-4),
        clip_norm=args.clip_norm, verbose=1,
        log_every=max(1, args.epochs // 20),
    )

    print("\n" + history.summary())
    print("\n" + history.plot(["loss", "val_loss"]))

    if dataset.X_test:
        print(_banner("Test set (used exactly once, at the end)"))
        results = model.evaluate(dataset.X_test, dataset.y_test)
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        preds = model.predict(dataset.X_test)
        print("\n" + format_confusion_matrix(
            confusion_matrix(preds, dataset.y_test),
            [str(c) for c in encoder.classes]))
        print("\n" + classification_report(preds, dataset.y_test,
                                           [str(c) for c in encoder.classes]))

    if args.save:
        model.save(args.save, scaler=dataset.scaler, encoder=encoder,
                   metadata={"features": feature_names,
                             "final_val_loss": history.last("val_loss")})
        print(f"\nSaved model + preprocessing to {args.save}")
    return 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def cmd_predict(args: argparse.Namespace) -> int:
    """Run inference with a saved model."""
    from .serialization import load_model

    model, extras = load_model(args.model, with_extras=True)
    scaler = extras.get("scaler")
    encoder = extras.get("encoder")

    print(f"Loaded {args.model}")
    print(model.summary())

    if args.input:
        rows = [[float(v) for v in chunk.split(",")]
                for chunk in args.input.split(";")]
    elif args.input_csv:
        from .data import load_csv
        rows, _, _ = load_csv(args.input_csv, target_column=None)
    else:
        print("\nGive --input '0,1' (semicolons separate samples) or --input-csv FILE")
        return 1

    raw_rows = [list(r) for r in rows]
    if scaler is not None:
        # Apply the exact transformation training used — without this the model
        # sees units it has never encountered.
        rows = scaler.transform(rows)
        print("\nApplied the saved scaler to the inputs")

    probabilities = model.predict_proba(rows)
    labels = model.predict_classes(rows, threshold=args.threshold)

    print(_banner("Predictions"))
    for i, (raw, probs, label) in enumerate(zip(raw_rows, probabilities, labels)):
        name = encoder.inverse_transform([label])[0] if encoder else label
        confidence = max(probs)
        print(f"\n  sample {i}: {raw}")
        print(f"    predicted class : {name}   (confidence {confidence:.4f})")
        print("    probabilities   : " +
              ", ".join(f"{j}={p:.4f}" for j, p in enumerate(probs)))
    return 0


# ---------------------------------------------------------------------------
# gradient-check
# ---------------------------------------------------------------------------

def cmd_gradient_check(args: argparse.Namespace) -> int:
    """Verify analytical gradients against finite differences."""
    from .layers import BatchNorm, Dense, Dropout
    from .model import Sequential
    from .numerical import check_gradients, gradient_check_report
    from .optimizers import SGD
    from .utils import set_seed

    set_seed(args.seed)
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y_bin = [[0.0], [1.0], [1.0], [0.0]]
    y_multi = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]

    configurations = [
        ("MSE + tanh + sigmoid", Sequential([
            Dense(2, 5, activation="tanh"),
            Dense(5, 1, activation="sigmoid")]), "mse", y_bin, 1e-6),
        ("BCE + relu", Sequential([
            Dense(2, 6, activation="relu"),
            Dense(6, 1, activation="sigmoid")]), "binary_cross_entropy", y_bin, 1e-5),
        ("softmax + categorical CE (fused)", Sequential([
            Dense(2, 6, activation="tanh"),
            Dense(6, 3, activation="softmax")]),
         "categorical_cross_entropy", y_multi, 1e-6),
        ("L1 + L2 regularization", Sequential([
            Dense(2, 5, activation="tanh", l2=0.01, l1=0.005),
            Dense(5, 1, activation="sigmoid", l2=0.01)]),
         "binary_cross_entropy", y_bin, 1e-5),
        ("BatchNorm", Sequential([
            Dense(2, 4, activation="identity"), BatchNorm(4),
            Dense(4, 1, activation="sigmoid")]), "mse", y_bin, 1e-6),
        ("ELU / GELU / Swish", Sequential([
            Dense(2, 4, activation="elu"),
            Dense(4, 4, activation="gelu"),
            Dense(4, 4, activation="swish"),
            Dense(4, 1, activation="sigmoid")]), "mse", y_bin, 1e-6),
        ("Huber + dropout", Sequential([
            Dense(2, 5, activation="tanh"), Dropout(0.3),
            Dense(5, 1, activation="identity")]), "huber", y_bin, 1e-6),
    ]

    print(_banner("NUMERICAL GRADIENT CHECK"))
    print("""
Backpropagation is verified against the definition of the derivative:

    numerical  =  [L(w + h) - L(w - h)] / (2h)        central difference
    rel_error  =  |analytical - numerical| / max(1, |a|, |n|)

A relative error below 1e-7 means the hand-derived calculus is correct.
""")

    if args.verbose:
        model = configurations[0][1]
        model.compile(loss=configurations[0][2], optimizer=SGD(0.1))
        print(gradient_check_report(model, X, configurations[0][3]))
        print()

    all_passed = True
    for name, model, loss, targets, tol in configurations:
        model.compile(loss=loss, optimizer=SGD(0.1))
        result = check_gradients(model, X, targets, tolerance=tol,
                                 max_checks_per_tensor=args.checks)
        status = "PASS" if result.passed else "FAIL"
        note = (f"  ({result.skipped_kinks} kinks skipped)"
                if result.skipped_kinks else "")
        print(f"  [{status}]  {name:<34} max rel err {result.max_error:.2e}"
              f"  n={result.num_checked}{note}")
        if not result.passed:
            all_passed = False
            for failure in result.failures[:3]:
                print(f"          {failure['param']}[{failure['index']}]: "
                      f"analytical={failure['analytical']:+.6e} "
                      f"numerical={failure['numerical']:+.6e}")

    print()
    if all_passed:
        print("All gradient checks passed — backpropagation is correct.")
        return 0
    print("Some gradient checks FAILED — backpropagation has a bug.")
    return 1


# ---------------------------------------------------------------------------
# test / demo
# ---------------------------------------------------------------------------

def cmd_test(args: argparse.Namespace) -> int:
    """Run the unit-test suite."""
    import unittest

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tests_dir = os.path.join(here, "tests")
    if not os.path.isdir(tests_dir):
        print(f"No tests directory found at {tests_dir}")
        return 1

    # unittest writes its report to stderr; print the banner there too so the
    # two stay in order when the output is piped.
    print(_banner(f"Running the test suite from {tests_dir}"), file=sys.stderr)
    sys.stderr.flush()
    loader = unittest.TestLoader()
    suite = loader.discover(tests_dir, pattern="test_*.py", top_level_dir=here)
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    return 0 if runner.run(suite).wasSuccessful() else 1


def cmd_demo(args: argparse.Namespace) -> int:
    """Walk one sample through the whole machine, printing every intermediate."""
    from .layers import Dense
    from .losses import BinaryCrossEntropy
    from .model import Sequential
    from .optimizers import SGD
    from .tensor import Tensor
    from .utils import set_seed

    set_seed(2)
    print(_banner("One training step, with every intermediate value shown"))
    print("""
Following the sample x = [0, 1] with target y = 1 through a 2 -> 3 -> 1
network. Every number below is computed by the equations in layers.py.
Watch the third hidden unit: relu drives it to 0, so its gradient is 0 too
and its incoming weights do not move this step. That is a "dead" unit.""")

    model = Sequential([
        Dense(2, 3, activation="relu", initialization="he"),
        Dense(3, 1, activation="sigmoid", initialization="xavier"),
    ])
    model.compile(loss="binary_cross_entropy", optimizer=SGD(0.5))

    x = Tensor([[0.0, 1.0]])
    y = Tensor([[1.0]])
    d1, d2 = model.layers[0], model.layers[1]

    def show(label: str, tensor: Tensor) -> None:
        print(f"  {label:<28} {tensor.shape}  {tensor.tolist()}")

    print("\nFORWARD")
    print("-" * 68)
    show("x (input)", x)
    show("W1", d1.W)
    show("b1", d1.b)
    z1 = x.matmul(d1.W)
    show("z1 = x @ W1", z1)
    a1 = model.layers[0].forward(x, training=True)
    show("z1 + b1 (pre-activation)", d1._pre_activation)
    show("a1 = relu(z1 + b1)", a1)
    show("W2", d2.W)
    p = model.layers[1].forward(a1, training=True)
    show("z2 = a1 @ W2 + b2", d2._pre_activation)
    show("p = sigmoid(z2)", p)

    loss_fn = BinaryCrossEntropy()
    loss = loss_fn.forward(p, y)
    print(f"\n  target y                     {y.tolist()}")
    print(f"  loss = -[y log p + (1-y) log(1-p)]  =  {loss:.6f}")

    print("\nBACKWARD")
    print("-" * 68)
    model.zero_grad()
    dp = loss_fn.backward(p, y)
    show("dL/dp", dp)
    dz2 = d2.activation.backward(dp)
    show("dL/dz2 = dL/dp * p(1-p)", dz2)
    da1 = d2.backward(dp)
    show("dL/dW2 = a1^T @ dL/dz2", d2.dW)
    show("dL/db2 = sum(dL/dz2)", d2.db)
    show("dL/da1 = dL/dz2 @ W2^T", da1)
    dx = d1.backward(da1)
    show("dL/dz1 = dL/da1 * (z1>0)", Tensor(
        [da1.data[i] * (1.0 if d1._pre_activation.data[i] > 0 else 0.0)
         for i in range(da1.size)], da1.shape))
    show("dL/dW1 = x^T @ dL/dz1", d1.dW)
    show("dL/db1", d1.b if d1.db is None else d1.db)
    show("dL/dx", dx)

    print("\nUPDATE  (SGD: W <- W - lr * dL/dW,  lr = 0.5)")
    print("-" * 68)
    before = d1.W.copy()
    model.optimizer.step(model.parameters(), model.gradients())
    show("W1 before", before)
    show("W1 after", d1.W)
    delta = [d1.W.data[i] - before.data[i] for i in range(before.size)]
    print(f"  change                       {[round(v, 6) for v in delta]}")

    new_loss = loss_fn.forward(model.forward(x, training=False), y)
    print(f"\n  loss before update: {loss:.6f}")
    print(f"  loss after update:  {new_loss:.6f}   "
          f"({'improved' if new_loss < loss else 'worse'})")
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scratch_nn",
        description="A neural network built from Python's standard library alone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python -m scratch_nn xor
  python -m scratch_nn demo
  python -m scratch_nn gradient-check
  python -m scratch_nn train --data data.csv --epochs 200 --save models/m.json
  python -m scratch_nn predict --model models/m.json --input "5.1,3.5,1.4,0.2"
  python -m scratch_nn test
""")
    parser.add_argument("--version", action="version",
                        version=f"scratch_nn {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_xor = sub.add_parser("xor", help="train a network on XOR")
    p_xor.add_argument("--epochs", type=int, default=500)
    p_xor.add_argument("--hidden", type=int, default=8)
    p_xor.add_argument("--lr", type=float, default=0.05)
    p_xor.add_argument("--optimizer", default="adam")
    p_xor.add_argument("--seed", type=int, default=42)
    p_xor.add_argument("--save", help="write the trained model to this path")
    p_xor.add_argument("--plot", action="store_true", help="show an ASCII loss curve")
    p_xor.set_defaults(func=cmd_xor)

    p_train = sub.add_parser("train", help="train on a CSV (or synthetic) dataset")
    p_train.add_argument("--data", help="path to a CSV file")
    p_train.add_argument("--target", default="-1",
                         help="target column index or name (default: last)")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--hidden", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=0.01)
    p_train.add_argument("--optimizer", default="adam")
    p_train.add_argument("--l2", type=float, default=0.0)
    p_train.add_argument("--dropout", type=float, default=0.0)
    p_train.add_argument("--clip-norm", type=float, default=None)
    p_train.add_argument("--patience", type=int, default=20)
    p_train.add_argument("--val-size", type=float, default=0.15)
    p_train.add_argument("--test-size", type=float, default=0.15)
    p_train.add_argument("--missing", default="error",
                         choices=["error", "zero", "mean"])
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--save", help="write the trained model to this path")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="run inference with a saved model")
    p_pred.add_argument("--model", required=True, help="path to a saved model JSON")
    p_pred.add_argument("--input", help="comma-separated features; ';' separates samples")
    p_pred.add_argument("--input-csv", help="a CSV of feature rows (no target column)")
    p_pred.add_argument("--threshold", type=float, default=0.5)
    p_pred.set_defaults(func=cmd_predict)

    p_grad = sub.add_parser("gradient-check",
                            help="verify backpropagation numerically")
    p_grad.add_argument("--checks", type=int, default=10,
                        help="entries probed per parameter tensor")
    p_grad.add_argument("--seed", type=int, default=42)
    p_grad.add_argument("--verbose", action="store_true",
                        help="print a full per-parameter table")
    p_grad.set_defaults(func=cmd_gradient_check)

    p_test = sub.add_parser("test", help="run the unit-test suite")
    p_test.add_argument("--verbose", action="store_true")
    p_test.set_defaults(func=cmd_test)

    p_demo = sub.add_parser(
        "demo", help="trace one sample through forward, backward and update")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    # `--target` accepts an index or a column name; normalise it here so the
    # command body does not have to.
    if getattr(args, "target", None) is not None:
        try:
            args.target = int(args.target)
        except (ValueError, TypeError):
            pass

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
