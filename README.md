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
python -m scratch_nn test             # 281 tests
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
| Metrics | accuracy, precision, recall, F1, confusion matrix, MAE/RMSE/R² |
| Verification | finite-difference gradient checking; 281 unit tests |

Requires Python 3.8+. `requirements.txt` is deliberately empty.
