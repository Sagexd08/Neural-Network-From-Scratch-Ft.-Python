"""
Evaluating a classifier: confusion matrix, confusion graph, and F1.

Run:  python examples/confusion_graph.py

Accuracy is the metric everyone reaches for first and the one that hides the
most.  On a dataset that is 99% class 0, a model that always answers 0 scores
99% accuracy while being completely useless.  This example trains a deliberately
imperfect three-class model and then looks at *where* it is wrong, which is the
question accuracy cannot answer.

It produces three views of the same result:

1. the confusion matrix as a text grid   - exact, good for a terminal
2. the confusion matrix as an SVG heatmap - scannable, good for a report
3. precision / recall / F1 per class      - the numbers behind the picture

The SVG is written by string formatting, with no plotting library, which is why
this file still runs with an empty requirements.txt.

It also regenerates ``reports/confusion_matrix.svg``, the image embedded in
README.md, so that figure is reproducible rather than a mystery artifact.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import scratch_nn as nn


REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports",
                           "confusion_matrix.svg")
CLASS_NAMES = ["class A", "class B", "class C"]


def main() -> int:
    # A fixed seed makes the figure in the README reproducible.
    nn.set_seed(7)

    # Deliberately overlapping blobs: a cleanly separable problem would give a
    # perfect diagonal and show nothing interesting.  cluster_std=2.2 puts the
    # classes close enough that the model must make real mistakes.
    X, y = nn.make_blobs(n_samples=600, n_features=2, centers=3, cluster_std=2.2)
    X_train, X_test, y_train, y_test = nn.train_test_split(
        X, y, test_size=0.3, stratify=True)

    # Fit the scaler on the training split only.  Fitting on everything would
    # leak test statistics into training and flatter the numbers below.
    scaler = nn.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = nn.Sequential([
        nn.Dense(2, 16, activation="relu"),
        nn.Dense(16, 3, activation="softmax"),
    ])
    model.compile(loss="categorical_cross_entropy",
                  optimizer=nn.Adam(0.03),
                  metrics=["accuracy"])
    model.fit(X_train, nn.one_hot(y_train, 3),
              epochs=120, batch_size=32, verbose=0)

    predictions = model.predict(X_test)
    matrix = nn.confusion_matrix(predictions, y_test)

    # 1. the text grid ------------------------------------------------------
    print(nn.format_confusion_matrix(matrix, CLASS_NAMES))
    print()

    # 2. the per-class numbers ---------------------------------------------
    print(nn.classification_report(predictions, y_test, CLASS_NAMES))
    print()

    accuracy = nn.accuracy(predictions, y_test)
    macro_f1 = nn.f1_score(predictions, y_test, average="macro")
    print(f"accuracy  {accuracy:.4f}")
    print(f"macro F1  {macro_f1:.4f}")
    print()

    # Accuracy and macro-F1 diverge exactly when the classes are handled
    # unevenly: macro-F1 weights every class the same regardless of size, so a
    # class the model quietly ignores drags it down while accuracy shrugs.
    if abs(accuracy - macro_f1) > 0.02:
        print("accuracy and macro F1 disagree -> the per-class errors are "
              "uneven; read the matrix above, not the accuracy alone.")
    else:
        print("accuracy and macro F1 agree -> errors are spread evenly "
              "across the classes.")
    print()

    # 3. the confusion graph ------------------------------------------------
    path = nn.save_confusion_matrix_svg(
        matrix, REPORT_PATH, class_names=CLASS_NAMES,
        title="Confusion matrix - 3-class blobs (test set)")
    print(f"confusion graph written to {os.path.normpath(path)}")
    print("open it in a browser: green = correct, red = misclassified, "
          "intensity = share of that true class.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
