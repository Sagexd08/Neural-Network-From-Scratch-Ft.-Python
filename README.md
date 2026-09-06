# Neural-Network-From-Scratch-Ft.-Python

A complete neural network framework built with **nothing but Python's standard
library** — no NumPy, no PyTorch, no TensorFlow, no autodiff.

Every matrix multiply is a readable loop. Every derivative was worked out by
hand and documented next to the code implementing it. Every gradient is
verified against finite differences to ~1e-11.

```bash
cd scratch_nn
python -m scratch_nn xor              # trains a network on XOR -> 4/4
python -m scratch_nn demo             # one sample, every intermediate value
python -m scratch_nn gradient-check   # proves backpropagation is correct
python -m scratch_nn test             # 315 tests
```

**Full documentation, including the mathematics from first principles, is in
[`scratch_nn/README.md`](scratch_nn/README.md).**

## What's here

| | |
|---|---|
| Tensor engine | N-d arrays on flat lists: matmul, broadcasting, reductions |
| Layers | Dense, Dropout, BatchNorm, Flatten, Residual |
| Activations | ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax, ELU, GELU, Swish |
| Losses | MSE, MAE, Binary CE, Categorical CE, Huber |
| Optimizers | SGD, Momentum, Nesterov, Adam, AMSGrad, RMSProp, AdaGrad |
| Training | mini-batch, early stopping, checkpointing, LR schedules, clipping |
| Data | CSV loading, leak-free scaling, one-hot encoding, stratified splits |
| Checkpoints | readable JSON, plus `.pkl` export; bit-for-bit round-trip |
| Metrics | accuracy, precision, recall, F1, confusion matrix + SVG confusion graph, MAE/RMSE/R² |
| Verification | finite-difference gradient checking; 315 unit tests |

Requires Python 3.8+. `requirements.txt` is deliberately empty.

## Evaluation

Accuracy alone hides the failure that matters: on a dataset that is 99% class
0, always answering 0 scores 99% accuracy while being useless. So the library
reports the confusion matrix and per-class F1 as well — and renders both as
figures, using SVG built by string formatting rather than a plotting library,
so the zero-dependency rule still holds.

Everything below is real output from
[`examples/confusion_graph.py`](scratch_nn/examples/confusion_graph.py), which
regenerates these figures from a fixed seed.

### Confusion matrix

```
Confusion matrix  (rows = actual, columns = predicted)

                  class A  class B  class C
actual class A         49        5        6
actual class B          7       47        6
actual class C          7       12       41
```

### Confusion graph

The same matrix as a heatmap. Diagonal cells are green and off-diagonal red,
so correctness reads as hue and magnitude as intensity — a healthy model is a
green stripe, and any bright red cell names the exact pair of classes being
confused.

<p align="center">
  <img src="scratch_nn/reports/confusion_matrix.svg" alt="Confusion matrix heatmap: a green diagonal with red off-diagonal cells showing which classes are confused" width="470">
</p>

Intensity is normalised **per row**, not globally. Rows are the actual
classes, so a row answers "of the true class-*i* examples, where did they
go?" — which is recall. Under global normalisation a large class saturates
every colour and a rare class stays invisible however badly it is classified.

### Precision, recall and F1

<p align="center">
  <img src="scratch_nn/reports/classification_report.svg" alt="Per-class precision, recall and F1 scores with proportional bars behind each F1 value" width="570">
</p>

Precision and recall trade against each other — predict positive for
everything and recall hits 1.0 while precision collapses. F1 is their
**harmonic** mean, which is the point: it is dominated by the smaller value,
so F1 is high only when both are.

| | precision | recall | arithmetic mean | **F1** |
|---|---|---|---|---|
| Predicts every sample positive | 0.25 | 1.00 | 0.63 — flattering | **0.40** |
| Never predicts positive | 0.00 | 0.00 | 0.00 | **0.00** |
| The model above (macro) | 0.76 | 0.76 | 0.76 | **0.76** |

```python
nn.f1_score(y_pred, y_true)                    # binary, positive class
nn.f1_score(y_pred, y_true, average="macro")   # multiclass, classes weighted equally

nn.save_confusion_matrix_svg(cm, "reports/confusion_matrix.svg")
nn.save_classification_report_svg(y_pred, y_true, "reports/report.svg")
```

The full derivation — why the harmonic mean, and why macro-averaging catches
what accuracy misses — is in
[the main README](scratch_nn/README.md#18-evaluation--beyond-accuracy).
