"""
A complete supervised-learning pipeline on a CSV dataset.

Run:  python examples/binary_classification.py

This walks the full lifecycle:

    generate CSV -> load -> split -> normalize -> train -> validate
                 -> evaluate on a held-out test set -> save -> reload -> predict

The dataset is a synthetic medical-screening problem: given four
measurements, predict whether a condition is present.  It is generated with
the standard library so the example needs no download, and written to CSV so
the loading path is genuinely exercised rather than bypassed.

Two things this example is careful about, because both are easy to get wrong
and both silently inflate your reported accuracy:

1. **The scaler is fitted on the training split only.**  Fitting it on all
   the data before splitting leaks the test set's distribution into training.
2. **The test set is touched exactly once**, at the very end.  Validation
   guides early stopping; if you also chose your stopping point using the
   test set, its score would no longer be an unbiased estimate.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from scratch_nn import (Adam, Dense, Dropout, LabelEncoder, Sequential,
                        StandardScaler, accuracy, classification_report,
                        confusion_matrix, f1_score, format_confusion_matrix,
                        load_csv, precision, recall, save_csv, set_seed,
                        train_val_test_split)
from scratch_nn.serialization import load_model
from scratch_nn.training import EarlyStopping
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


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "screening.csv")
MODEL_PATH = os.path.join(HERE, "..", "models", "screening.json")

FEATURES = ["age", "biomarker", "pressure", "score"]


def generate_dataset(path, n_samples=600, seed=7):
    """Write a synthetic screening dataset to CSV.

    The label depends nonlinearly on the features (there is an interaction
    term and a threshold), so a linear model cannot do well and the hidden
    layers have something real to learn.  Label noise is added so that 100%
    accuracy is impossible — which makes the overfitting story visible.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n_samples):
        age = rng.gauss(50, 15)
        biomarker = rng.gauss(0, 1)
        pressure = rng.gauss(120, 18)
        score = rng.uniform(0, 10)

        # A deliberately nonlinear decision rule.
        risk = (0.03 * (age - 50)
                + 1.2 * biomarker
                + 0.02 * (pressure - 120)
                + 0.15 * score
                + 0.8 * biomarker * (score / 10.0)      # interaction
                - 1.0)
        probability = 1.0 / (1.0 + math.exp(-risk))
        label = 1 if rng.random() < probability else 0   # label noise
        rows.append([round(age, 2), round(biomarker, 4),
                     round(pressure, 2), round(score, 3), label])

    save_csv(path, rows, header=FEATURES + ["diagnosis"])
    return path


def main():
    set_seed(42)

    # ---- 1. data ---------------------------------------------------------
    rule("1. DATA")
    generate_dataset(DATA_PATH)
    print(f"\nWrote a synthetic dataset to {os.path.normpath(DATA_PATH)}")

    X, y_raw, feature_names = load_csv(DATA_PATH, target_column="diagnosis")
    print(f"Loaded {len(X)} samples with {len(feature_names)} features: "
          f"{feature_names}")

    positives = sum(1 for v in y_raw if float(v) == 1.0)
    print(f"Class balance: {positives} positive, {len(y_raw) - positives} negative "
          f"({positives / len(y_raw):.1%} positive)")

    print("\nFirst three rows, unscaled:")
    for i in range(3):
        pairs = ", ".join(f"{n}={v:g}" for n, v in zip(feature_names, X[i]))
        print(f"  {pairs}  ->  diagnosis={int(float(y_raw[i]))}")

    # ---- 2. split --------------------------------------------------------
    rule("2. SPLIT  (before any preprocessing)")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y_raw, val_size=0.15, test_size=0.15, stratify=True)
    print(f"\n  train      {len(X_train):>4}  fits the weights")
    print(f"  validation {len(X_val):>4}  chooses when to stop")
    print(f"  test       {len(X_test):>4}  touched once, at the very end")

    # ---- 3. preprocessing ------------------------------------------------
    rule("3. PREPROCESSING  (fitted on the training split only)")
    scaler = StandardScaler().fit(X_train)
    print("\nPer-feature statistics learned from the training data:")
    print(f"  {'feature':<12} {'mean':>10} {'std':>10}")
    print("  " + "-" * 34)
    for name, mean, std in zip(feature_names, scaler.mean, scaler.std):
        print(f"  {name:<12} {mean:>10.3f} {std:>10.3f}")

    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("\nWhy this matters: 'pressure' spans roughly 120 +/- 18 while")
    print("'biomarker' spans 0 +/- 1. Without scaling, the gradients for")
    print("pressure's weights would be ~100x larger, so any learning rate")
    print("suited to one feature would diverge or stall on the other.")

    print("\nThe same three rows, standardized:")
    for i in range(3):
        print("  " + ", ".join(f"{v:+.3f}" for v in X_train_s[i]))

    encoder = LabelEncoder().fit(y_raw)
    y_train_e = [[float(v)] for v in encoder.transform(y_train)]
    y_val_e = [[float(v)] for v in encoder.transform(y_val)]
    y_test_e = [[float(v)] for v in encoder.transform(y_test)]

    # ---- 4. model --------------------------------------------------------
    rule("4. MODEL")
    model = Sequential([
        Dense(len(feature_names), 16, activation="relu",
              initialization="he", l2=1e-4),
        Dropout(0.2),
        Dense(16, 8, activation="relu", initialization="he", l2=1e-4),
        Dense(8, 1, activation="sigmoid", initialization="xavier"),
    ])
    model.compile(loss="binary_cross_entropy", optimizer=Adam(0.005),
                  metrics=["accuracy"])
    print()
    print(model.summary())
    print("\nRegularization in use:")
    print("  L2 (1e-4) shrinks weights toward zero, preferring many small")
    print("     weights over a few large ones, which generalizes better.")
    print("  Dropout (0.2) randomly zeroes a fifth of the first layer's")
    print("     activations each step, so no unit can rely on any other.")

    # ---- 5. training -----------------------------------------------------
    rule("5. TRAINING")
    print()
    stopper = EarlyStopping(monitor="val_loss", patience=25, min_delta=1e-4,
                            restore_best_weights=True, verbose=True)
    history = model.fit(
        X_train_s, y_train_e,
        epochs=400, batch_size=32,
        validation_data=(X_val_s, y_val_e),
        early_stopping=stopper,
        clip_norm=5.0,
        verbose=1, log_every=20,
    )

    rule("Learning curves")
    print()
    print(history.plot(["loss", "val_loss"], height=11, width=58))
    print()
    print(history.summary())

    gap = history.last("val_loss") - history.last("loss")
    print(f"\nGeneralization gap (val_loss - train_loss): {gap:+.4f}")
    print("A gap near zero means the model generalizes; a large positive gap")
    print("means it has memorised the training set. Early stopping restored")
    print(f"the weights from epoch {history.best_epoch}, where validation loss")
    print("was at its minimum.")

    # ---- 6. evaluation ---------------------------------------------------
    rule("6. TEST SET  (used exactly once)")
    results = model.evaluate(X_test_s, y_test_e)
    print()
    for key, value in results.items():
        print(f"  {key:<12} {value:.4f}")

    predictions = model.predict(X_test_s)
    print()
    print(format_confusion_matrix(
        confusion_matrix(predictions, y_test_e), ["negative", "positive"]))
    print()
    print(classification_report(predictions, y_test_e, ["negative", "positive"]))

    print("\nReading these numbers:")
    print(f"  precision {precision(predictions, y_test_e):.3f} — of the cases")
    print("     flagged positive, this fraction really were.")
    print(f"  recall    {recall(predictions, y_test_e):.3f} — of the real")
    print("     positives, this fraction were found.")
    print("  For a medical screen you would tune the decision threshold to")
    print("  favour recall, since a missed case costs more than a false alarm.")

    # ---- 7. threshold sensitivity ----------------------------------------
    rule("7. THE DECISION THRESHOLD IS A CHOICE")
    print("\nThe network outputs a probability. Turning it into a yes/no")
    print("answer requires a threshold, and that threshold trades precision")
    print("against recall:\n")
    print(f"  {'threshold':>10} {'accuracy':>10} {'precision':>11} {'recall':>8} {'F1':>8}")
    print("  " + "-" * 51)
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        print(f"  {threshold:>10.1f} "
              f"{accuracy(predictions, y_test_e, threshold):>10.3f} "
              f"{precision(predictions, y_test_e, threshold=threshold):>11.3f} "
              f"{recall(predictions, y_test_e, threshold=threshold):>8.3f} "
              f"{f1_score(predictions, y_test_e, threshold=threshold):>8.3f}")

    # ---- 8. save and reload ----------------------------------------------
    rule("8. SAVE, RELOAD, AND INFER")
    model.save(MODEL_PATH, scaler=scaler, encoder=encoder,
               metadata={"features": feature_names,
                         "test_accuracy": results["accuracy"],
                         "best_epoch": history.best_epoch})
    size = os.path.getsize(MODEL_PATH)
    print(f"\nSaved to {os.path.normpath(MODEL_PATH)}  ({size:,} bytes of JSON)")
    print("The file contains the architecture, every weight, the optimizer")
    print("state, and the fitted scaler — everything inference needs.")

    reloaded, extras = load_model(MODEL_PATH, with_extras=True)
    reloaded_predictions = reloaded.predict(X_test_s)
    identical = reloaded_predictions.data == predictions.data
    print(f"\nPredictions identical after reload: {identical}")

    # ---- 9. inference on new data ----------------------------------------
    rule("9. INFERENCE ON NEW, UNSEEN PATIENTS")
    new_patients = [
        [68.0, 1.8, 145.0, 8.5],    # older, high biomarker -> expect positive
        [32.0, -1.2, 110.0, 2.0],   # younger, low biomarker -> expect negative
        [50.0, 0.0, 120.0, 5.0],    # exactly average -> expect uncertain
    ]
    saved_scaler = extras["scaler"]
    saved_encoder = extras["encoder"]

    # The saved scaler must be applied: the model has only ever seen
    # standardized units, so feeding it raw values would be meaningless.
    scaled = saved_scaler.transform(new_patients)
    probabilities = reloaded.predict(scaled)

    print()
    for patient, raw in zip(new_patients, range(len(new_patients))):
        p = probabilities.data[raw]
        label = 1 if p >= 0.5 else 0
        name = saved_encoder.inverse_transform([label])[0]
        confidence = "confident" if abs(p - 0.5) > 0.25 else "uncertain"
        pairs = ", ".join(f"{n}={v:g}" for n, v in zip(feature_names, patient))
        print(f"  {pairs}")
        print(f"    P(positive) = {p:.4f}  ->  diagnosis {int(float(name))} "
              f"({confidence})\n")

    rule("PIPELINE COMPLETE")
    print(f"\n  test accuracy : {results['accuracy']:.1%}")
    print(f"  epochs run    : {history.epochs} (best was {history.best_epoch})")
    print(f"  model file    : {os.path.normpath(MODEL_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
