"""
Multiclass classification with softmax and categorical cross-entropy.

Run:  python examples/multiclass_classification.py

Binary classification uses one sigmoid output and asks "how likely is
positive?".  With three or more mutually exclusive classes you instead want a
whole probability *distribution* over the classes, which is what softmax
produces:

    p_i = e^(z_i) / sum_j e^(z_j)          every p_i > 0, and they sum to 1

Paired with categorical cross-entropy, ``L = -log(p of the true class)``, the
two functions differentiate together into something remarkably clean:

    dL/dz = p - y                          "predicted minus actual"

This example verifies that identity numerically, then trains on a three-class
problem and inspects where the model is uncertain.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from scratch_nn import (Adam, Dense, Sequential, StandardScaler, Tensor,
                        accuracy, classification_report, confusion_matrix,
                        f1_score, format_confusion_matrix, make_blobs, one_hot,
                        set_seed, train_val_test_split)
from scratch_nn.activations import Softmax
from scratch_nn.losses import CategoricalCrossEntropy
from scratch_nn.numerical import check_gradients
from scratch_nn.training import EarlyStopping, LearningRateScheduler, cosine_decay
from scratch_nn.utils import safe_text

_builtin_print = print


def print(*args, **kwargs):  # noqa: A001 - deliberate shadow of the builtin
    """``print`` that degrades to ASCII on a legacy Windows code page."""
    _builtin_print(*[safe_text(a) if isinstance(a, str) else a for a in args],
                   **kwargs)


def rule(title=""):
    print(f"\n{'=' * 70}")
    if title:
        print(title)
        print("=" * 70)


CLASS_NAMES = ["setosa-like", "versicolor-like", "virginica-like"]


def demonstrate_softmax():
    """Show the numerical-stability trick and the fused gradient identity."""
    rule("1. HOW SOFTMAX WORKS")

    logits = Tensor([[1.0, 2.0, 3.0]])
    probabilities = Softmax().forward(logits)
    print("\nRaw scores (logits) out of the final Dense layer:")
    print(f"  {[round(v, 2) for v in logits.data]}")
    print("\nAfter softmax:")
    print(f"  {[round(v, 4) for v in probabilities.data]}")
    print(f"  they sum to {sum(probabilities.data):.10f}")

    print("\nWhy the max is subtracted first")
    print("-" * 46)
    print("The naive formula computes e^z directly. For a logit of 1000,")
    print("e^1000 overflows to infinity, and inf/inf is nan. Because softmax")
    print("is unchanged by shifting every input by a constant, subtracting")
    print("the row maximum makes the largest exponent e^0 = 1 exactly:")
    big = Softmax().forward(Tensor([[1000.0, 1001.0, 1002.0]]))
    print(f"\n  softmax([1000, 1001, 1002]) = {[round(v, 4) for v in big.data]}")
    print(f"  all finite: {all(math.isfinite(v) for v in big.data)}")
    print("  identical to softmax([1, 2, 3]) above, as shift-invariance requires.")

    print("\nThe fused gradient")
    print("-" * 46)
    p = Tensor([[0.7, 0.2, 0.1]])
    y = Tensor([[1.0, 0.0, 0.0]])
    fused = CategoricalCrossEntropy().backward_fused(p, y)
    print("With p = [0.7, 0.2, 0.1] and the true class 0, i.e. y = [1, 0, 0]:")
    print(f"\n  dL/dz = p - y = {[round(v, 4) for v in fused.data]}")
    print("\nThe true class gets a negative gradient (push its score up); the")
    print("others get positive ones (push their scores down). No division and")
    print("no exponentials appear, so nothing can overflow.")


def main():
    set_seed(42)
    demonstrate_softmax()

    # ---- 2. data ---------------------------------------------------------
    rule("2. A THREE-CLASS DATASET")
    X, y_labels = make_blobs(n_samples=450, n_features=4, centers=3,
                             cluster_std=1.6)
    print(f"\n{len(X)} samples, {len(X[0])} features, 3 classes")
    for c in range(3):
        count = sum(1 for v in y_labels if v == c)
        print(f"  class {c} ({CLASS_NAMES[c]}): {count} samples")

    # One-hot encoding: an integer label would falsely imply that class 2 is
    # "between" 1 and 3 and closer to 1 than to 0. One-hot places every class
    # at an equal distance from every other.
    Y = one_hot(y_labels, 3)
    print(f"\nLabel {y_labels[0]} becomes the one-hot row "
          f"{[int(v) for v in Y[0]]}")

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, Y, val_size=0.15, test_size=0.15, stratify=True)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    print(f"\nSplit: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test"
          f"  (scaler fitted on train only)")

    # ---- 3. model --------------------------------------------------------
    rule("3. MODEL")
    model = Sequential([
        Dense(4, 24, activation="relu", initialization="he"),
        Dense(24, 12, activation="relu", initialization="he"),
        Dense(12, 3, activation="softmax", initialization="xavier"),
    ])
    model.compile(loss="categorical_cross_entropy", optimizer=Adam(0.01),
                  metrics=["accuracy"])
    print()
    print(model.summary())
    print(f"\nFused softmax + cross-entropy gradient active: "
          f"{model._fused_softmax_ce}")

    rule("Verifying the gradients before training")
    result = check_gradients(model, X_train_s[:8], y_train[:8], tolerance=1e-5)
    print()
    print(result.summary())
    if not result.passed:
        return 1

    # ---- 4. training -----------------------------------------------------
    rule("4. TRAINING  (with a cosine learning-rate schedule)")
    print("\nThe learning rate follows a cosine from its full value down to")
    print("near zero: large exploratory steps early, tiny settling steps late.")
    print()
    history = model.fit(
        X_train_s, y_train,
        epochs=120, batch_size=32,
        validation_data=(X_val_s, y_val),
        callbacks=[LearningRateScheduler(cosine_decay(1.0, 120, 0.02))],
        early_stopping=EarlyStopping(monitor="val_loss", patience=30,
                                     min_delta=1e-4, verbose=True),
        verbose=1, log_every=10,
    )

    rule("Learning curves")
    print()
    print(history.plot(["loss", "val_loss"], height=10, width=56))
    print()
    print(history.plot(["accuracy", "val_accuracy"], height=10, width=56))
    print()
    print(history.summary())

    # ---- 5. evaluation ---------------------------------------------------
    rule("5. TEST SET")
    results = model.evaluate(X_test_s, y_test)
    print()
    for key, value in results.items():
        print(f"  {key:<12} {value:.4f}")

    predictions = model.predict(X_test_s)
    print()
    print(format_confusion_matrix(confusion_matrix(predictions, y_test),
                                  CLASS_NAMES))
    print()
    print(classification_report(predictions, y_test, CLASS_NAMES))
    print(f"\n  macro F1: {f1_score(predictions, y_test, average='macro'):.4f}")
    print("  Macro averaging weights every class equally, so a class the")
    print("  model ignores cannot be hidden by good scores on the others.")

    # ---- 6. confidence ---------------------------------------------------
    rule("6. WHAT THE PROBABILITIES MEAN")
    print("\nSoftmax gives a full distribution, not just a label, so the model")
    print("can tell you how sure it is. Sorting the test set by confidence:\n")

    rows = []
    for i in range(predictions.shape[0]):
        probs = [predictions.data[i * 3 + c] for c in range(3)]
        predicted = max(range(3), key=lambda c: probs[c])
        actual = max(range(3), key=lambda c: y_test[i][c])
        rows.append((max(probs), predicted, actual, probs))
    rows.sort(reverse=True)

    def show(entry):
        confidence, predicted, actual, probs = entry
        mark = "ok   " if predicted == actual else "WRONG"
        distribution = "  ".join(f"{p:.3f}" for p in probs)
        return (f"    {mark}  predicted {predicted}  actual {actual}   "
                f"[{distribution}]")

    print("  Most confident:")
    for entry in rows[:3]:
        print(show(entry))
    print("\n  Least confident:")
    for entry in rows[-3:]:
        print(show(entry))

    errors = [e for e in rows if e[1] != e[2]]
    if errors:
        mean_error_confidence = sum(e[0] for e in errors) / len(errors)
        correct = [e for e in rows if e[1] == e[2]]
        mean_correct_confidence = sum(e[0] for e in correct) / len(correct)
        print(f"\n  mean confidence when correct: {mean_correct_confidence:.3f}")
        print(f"  mean confidence when wrong:   {mean_error_confidence:.3f}")
        print("\n  A well-calibrated model is less confident when it is wrong,")
        print("  which is what lets you route uncertain cases to a human.")
    else:
        print("\n  No errors on the test set.")

    rule("DONE")
    print(f"\n  test accuracy : {results['accuracy']:.1%}")
    print(f"  epochs run    : {history.epochs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
