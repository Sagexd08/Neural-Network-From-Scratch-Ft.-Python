"""
XOR — the problem that made hidden layers necessary.

Run:  python examples/xor.py

In 1969 Minsky and Papert proved that a perceptron with no hidden layer
cannot compute XOR.  The argument is one sentence: a single layer draws one
straight line through the input space, and no straight line separates
{(0,1), (1,0)} from {(0,0), (1,1)}.  The result was read as a verdict on
neural networks generally, and funding largely dried up for a decade.

The resolution is a hidden layer.  It re-represents the inputs in a space
where the classes *are* linearly separable, and the output layer then draws
its line there.  This script demonstrates both halves of that story: the
single-layer model fails, and the two-hidden-layer model succeeds.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from scratch_nn import (Adam, Dense, Sequential, set_seed, xor_dataset)
from scratch_nn.numerical import check_gradients
from scratch_nn.utils import safe_text

_builtin_print = print


def print(*args, **kwargs):  # noqa: A001 - deliberate shadow of the builtin
    """``print`` that degrades to ASCII on a legacy Windows code page.

    A cp1252 console cannot encode an em-dash or a block character and raises
    ``UnicodeEncodeError``; :func:`~scratch_nn.utils.safe_text` substitutes
    ASCII equivalents there and does nothing on a UTF-8 terminal.
    """
    _builtin_print(*[safe_text(a) if isinstance(a, str) else a for a in args],
                   **kwargs)


def rule(title=""):
    print(f"\n{'=' * 66}")
    if title:
        print(title)
        print("=" * 66)


def main():
    set_seed(42)
    X, y = xor_dataset()

    rule("THE DATA")
    print("\n   a    b  |  a XOR b")
    print("  ----------|---------")
    for row, target in zip(X, y):
        print(f"   {int(row[0])}    {int(row[1])}  |     {int(target[0])}")

    # ---- 1. the failure a single layer cannot avoid ----------------------
    rule("1. NO HIDDEN LAYER  (expected to fail)")
    flat = Sequential([Dense(2, 1, activation="sigmoid")])
    flat.compile(loss="binary_cross_entropy", optimizer=Adam(0.1),
                 metrics=["accuracy"])
    flat_history = flat.fit(X, y, epochs=500, batch_size=4, verbose=0)
    print(f"\nAfter 500 epochs: loss={flat_history.last('loss'):.4f}  "
          f"accuracy={flat_history.last('accuracy'):.0%}")
    print("The loss plateaus near 0.69 = ln(2), which is exactly the loss of")
    print("guessing 50/50 on every input. The model has learned nothing,")
    print("because there is nothing a single line can learn here.")

    # ---- 2. what a hidden layer buys -------------------------------------
    rule("2. TWO HIDDEN LAYERS")
    model = Sequential([
        Dense(2, 8, activation="tanh", initialization="xavier"),
        Dense(8, 8, activation="tanh", initialization="xavier"),
        Dense(8, 1, activation="sigmoid", initialization="xavier"),
    ])
    model.compile(loss="binary_cross_entropy", optimizer=Adam(0.05),
                  metrics=["accuracy"])
    print()
    print(model.summary())

    # Before training a single step, confirm the gradients are correct.
    # If backpropagation were wrong, everything after this point would be
    # meaningless, so it is worth 200 milliseconds to be sure.
    rule("Verifying backpropagation before training")
    result = check_gradients(model, X, y, tolerance=1e-6)
    print()
    print(result.summary())
    if not result.passed:
        print("\nGradients are wrong; aborting.")
        return 1

    rule("Training")
    print()
    history = model.fit(X, y, epochs=500, batch_size=4, verbose=1, log_every=50)

    # ---- 3. results -------------------------------------------------------
    rule("3. RESULTS")
    predictions = model.predict(X)
    print("\n  input      probability   predicted   true")
    print("  " + "-" * 45)
    correct = 0
    for i, (row, target) in enumerate(zip(X, y)):
        p = predictions.data[i]
        label = 1 if p >= 0.5 else 0
        truth = int(target[0])
        correct += label == truth
        print(f"  {int(row[0])} XOR {int(row[1])}  ->  {p:>9.4f}   "
              f"{label:^9}   {truth:^4}")
    print(f"\n  accuracy: {correct}/4")

    rule("The loss curve")
    print()
    print(history.bars("loss", samples=12))
    print()
    print(history.plot(["loss"], height=10, width=56))

    # ---- 4. what the hidden layer actually learned ------------------------
    rule("4. WHAT THE HIDDEN LAYER LEARNED")
    print("\nEach hidden unit computes tanh(w1*a + w2*b + bias) — one soft")
    print("line through the input space. The output layer combines them.")
    print("Here is what the first hidden layer outputs for each input:\n")
    hidden = model.layers[0].forward(
        __import__("scratch_nn").Tensor(X), training=False)
    print("   input   |  first 4 hidden units")
    print("  ---------|" + "-" * 40)
    for i, row in enumerate(X):
        values = [hidden.data[i * 8 + j] for j in range(4)]
        formatted = "  ".join(f"{v:+.3f}" for v in values)
        print(f"   ({int(row[0])}, {int(row[1])})  |  {formatted}")
    print("\nRows 2 and 3 (the XOR=1 cases) now sit on the same side of a")
    print("boundary that rows 1 and 4 do not — separable by a single line,")
    print("which is precisely what the output layer needs.")

    rule("Decision boundary over the unit square")
    print()
    print(decision_boundary(model))

    return 0 if correct == 4 else 1


def decision_boundary(model, size=21):
    """Sample the model over [0,1]^2 and render it as a character grid."""
    from scratch_nn import Tensor
    grid_points = []
    for r in range(size):
        b = 1.0 - r / (size - 1)
        for c in range(size):
            a = c / (size - 1)
            grid_points.append([a, b])
    probabilities = model.predict(Tensor(grid_points)).data

    shades = " .:-=+*#%@"
    lines = []
    for r in range(size):
        row = "".join(
            shades[min(len(shades) - 1, int(probabilities[r * size + c] * len(shades)))]
            for c in range(size))
        lines.append(f"  {row}")
    body = "\n".join(lines)
    return (f"  b=1 (top) to b=0 (bottom), a=0 (left) to a=1 (right)\n"
            f"  ' '=P(1) near 0, '@'=P(1) near 1\n\n{body}\n\n"
            f"  The two bright corners are (0,1) and (1,0) — XOR = 1.")


if __name__ == "__main__":
    sys.exit(main())
