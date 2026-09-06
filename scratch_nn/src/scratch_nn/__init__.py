"""
scratch_nn
==========

A neural network framework written entirely with Python's standard library.

No NumPy, no PyTorch, no TensorFlow — every matrix multiply, every derivative
and every optimizer update is a Python loop you can read.

Quick start
-----------

>>> from scratch_nn import Sequential, Dense, Adam, set_seed
>>> set_seed(42)
>>> model = Sequential([
...     Dense(2, 8, activation="relu"),
...     Dense(8, 1, activation="sigmoid"),
... ])
>>> model.compile(loss="binary_cross_entropy", optimizer=Adam(0.01),
...               metrics=["accuracy"])
>>> history = model.fit(X, y, epochs=500, batch_size=4)
>>> model.predict([[0, 1]])

Where to look
-------------
``tensor.py``          N-d arrays over flat Python lists; matmul, broadcasting
``activations.py``     ReLU, sigmoid, tanh, softmax and their derivatives
``layers.py``          Dense, Dropout, BatchNorm — and the backprop derivation
``losses.py``          MSE, BCE, categorical cross-entropy
``optimizers.py``      SGD, Momentum, Adam, RMSProp, AdaGrad
``model.py``           Sequential: the forward and backward orchestration
``numerical.py``       Finite-difference gradient checking (the correctness proof)
``training.py``        The training loop, early stopping, LR schedules
``data.py``            CSV loading, splitting, scaling, one-hot encoding
``metrics.py``         Accuracy, precision, recall, F1, confusion graph, R^2
``serialization.py``   JSON checkpoints you can read in a text editor (or .pkl)
"""

from .activations import (ACTIVATIONS, ELU, GELU, Activation, Identity, LeakyReLU,
                          ReLU, Sigmoid, Softmax, Swish, Tanh, get_activation)
from .data import (Dataset, LabelEncoder, MinMaxScaler, StandardScaler,
                   batch_iterator, from_one_hot, load_csv, make_blobs,
                   make_classification, make_moons, make_regression, one_hot,
                   save_csv, shuffle_data, train_test_split,
                   train_val_test_split, xor_dataset)
from .initialization import (INITIALIZERS, get_initializer, he_normal_init,
                             he_uniform_init, lecun_normal_init,
                             xavier_normal_init, xavier_uniform_init, zeros_init)
from .layers import Activation_, BatchNorm, Dense, Dropout, Flatten, Layer, Residual
from .losses import (LOSSES, BinaryCrossEntropy, CategoricalCrossEntropy, Huber,
                     Loss, MeanAbsoluteError, MeanSquaredError, get_loss)
from .metrics import (accuracy, classification_report, confusion_matrix,
                      confusion_matrix_svg, f1_score,
                      format_confusion_matrix, mean_absolute_error,
                      mean_squared_error, precision, r_squared, recall,
                      regression_report, root_mean_squared_error,
                      save_confusion_matrix_svg)
from .model import Model, Sequential
from .numerical import (GradientCheckResult, check_gradients,
                        check_layer_gradients, gradient_check_report,
                        numerical_gradient, relative_error)
from .optimizers import (OPTIMIZERS, SGD, AdaGrad, Adam, Momentum, Optimizer,
                         RMSProp, get_optimizer)
from .serialization import (load_model, load_pickle, save_model,
                            save_pickle)
from .tensor import (Tensor, TensorShapeError, arange, eye, from_flat, full,
                     ones, rand, randn, stack_rows, zeros)
from .training import (Callback, EarlyStopping, History, LearningRateScheduler,
                       ModelCheckpoint, cosine_decay, evaluate,
                       exponential_decay, fit, step_decay)
from .utils import (EPS, ascii_plot, bar_chart, clip_gradient, get_rng,
                    gradient_norm, is_finite, parameter_norm, progress_bar,
                    set_seed)

__version__ = "1.0.0"

__all__ = [
    # tensor
    "Tensor", "TensorShapeError", "zeros", "ones", "full", "eye", "rand",
    "randn", "arange", "from_flat", "stack_rows",
    # activations
    "Activation", "Identity", "ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax",
    "ELU", "GELU", "Swish", "get_activation", "ACTIVATIONS",
    # initialization
    "get_initializer", "INITIALIZERS", "zeros_init", "xavier_uniform_init",
    "xavier_normal_init", "he_uniform_init", "he_normal_init", "lecun_normal_init",
    # layers
    "Layer", "Dense", "Dropout", "Flatten", "BatchNorm", "Residual", "Activation_",
    # losses
    "Loss", "MeanSquaredError", "MeanAbsoluteError", "BinaryCrossEntropy",
    "CategoricalCrossEntropy", "Huber", "get_loss", "LOSSES",
    # optimizers
    "Optimizer", "SGD", "Momentum", "Adam", "RMSProp", "AdaGrad",
    "get_optimizer", "OPTIMIZERS",
    # model
    "Sequential", "Model",
    # training
    "fit", "evaluate", "History", "Callback", "EarlyStopping", "ModelCheckpoint",
    "LearningRateScheduler", "step_decay", "exponential_decay", "cosine_decay",
    # data
    "Dataset", "load_csv", "save_csv", "train_test_split", "train_val_test_split",
    "shuffle_data", "batch_iterator", "StandardScaler", "MinMaxScaler",
    "LabelEncoder", "one_hot", "from_one_hot", "make_classification",
    "make_moons", "make_blobs", "make_regression", "xor_dataset",
    # metrics
    "accuracy", "confusion_matrix", "precision", "recall", "f1_score",
    "classification_report", "format_confusion_matrix",
    "confusion_matrix_svg", "save_confusion_matrix_svg", "mean_absolute_error",
    "mean_squared_error", "root_mean_squared_error", "r_squared",
    "regression_report",
    # gradient checking
    "check_gradients", "check_layer_gradients", "gradient_check_report",
    "numerical_gradient", "relative_error", "GradientCheckResult",
    # serialization
    "save_model", "load_model", "save_pickle", "load_pickle",
    # utils
    "set_seed", "get_rng", "is_finite", "gradient_norm", "parameter_norm",
    "clip_gradient", "ascii_plot", "bar_chart", "progress_bar", "EPS",
    "__version__",
]
