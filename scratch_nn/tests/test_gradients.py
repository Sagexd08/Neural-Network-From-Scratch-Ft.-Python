"""Numerical gradient checking — the correctness proof for backpropagation.

If any test in this file fails, backpropagation is wrong somewhere and every
model this library trains is silently suboptimal.  These are the tests that
matter most.

Each case builds a model, computes gradients analytically via
``model.backward()``, then independently estimates the same gradients with
central differences of the loss, and demands agreement to ~1e-6 relative
error.
"""

import unittest

from scratch_nn.layers import BatchNorm, Dense, Dropout, Flatten, Residual
from scratch_nn.model import Sequential
from scratch_nn.numerical import (check_gradients, check_layer_gradients,
                                  numerical_gradient, relative_error)
from scratch_nn.optimizers import SGD
from scratch_nn.tensor import Tensor
from scratch_nn.utils import set_seed

X_BINARY = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y_BINARY = [[0.0], [1.0], [1.0], [0.0]]
Y_MULTI = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]


class TestRelativeError(unittest.TestCase):

    def test_identical_values(self):
        self.assertEqual(relative_error(1.0, 1.0), 0.0)

    def test_scales_with_magnitude(self):
        # Both differ by 1e-6 absolutely, but the metric floors the
        # denominator at 1 so tiny gradients do not produce false alarms.
        self.assertAlmostEqual(relative_error(1e-9, 2e-9), 1e-9)
        self.assertLess(relative_error(1000.0, 1000.001), 1e-5)

    def test_detects_a_wrong_sign(self):
        self.assertGreater(relative_error(1.0, -1.0), 0.5)


class TestNumericalGradient(unittest.TestCase):

    def test_estimates_a_known_derivative(self):
        # f(w) = w^2 has f'(3) = 6.
        param = Tensor([3.0])
        value = numerical_gradient(lambda: param.data[0] ** 2, param, 0)
        self.assertAlmostEqual(value, 6.0, places=6)

    def test_restores_the_parameter(self):
        param = Tensor([3.0])
        numerical_gradient(lambda: param.data[0] ** 2, param, 0)
        self.assertEqual(param.data[0], 3.0)

    def test_restores_even_when_the_loss_raises(self):
        param = Tensor([3.0])

        def broken():
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            numerical_gradient(broken, param, 0)
        self.assertEqual(param.data[0], 3.0)


class TestModelGradients(unittest.TestCase):
    """Each case must pass, or backpropagation is broken for that path."""

    def _check(self, model, X, y, loss, tolerance=1e-6):
        model.compile(loss=loss, optimizer=SGD(0.01))
        result = check_gradients(model, X, y, tolerance=tolerance,
                                 max_checks_per_tensor=12, seed=0)
        self.assertTrue(
            result.passed,
            f"gradient check failed (max rel err {result.max_error:.3e}):\n"
            f"{result.summary()}")
        return result

    def test_mse_with_tanh(self):
        set_seed(1)
        self._check(Sequential([
            Dense(2, 5, activation="tanh"),
            Dense(5, 1, activation="identity")]), X_BINARY, Y_BINARY, "mse")

    def test_binary_cross_entropy_with_sigmoid(self):
        set_seed(2)
        self._check(Sequential([
            Dense(2, 6, activation="tanh"),
            Dense(6, 1, activation="sigmoid")]),
            X_BINARY, Y_BINARY, "binary_cross_entropy")

    def test_softmax_categorical_cross_entropy_fused(self):
        set_seed(3)
        model = Sequential([
            Dense(2, 6, activation="tanh"),
            Dense(6, 3, activation="softmax")])
        model.compile(loss="categorical_cross_entropy", optimizer=SGD(0.01))
        # Confirm the fused dZ = p - y path is what is being exercised.
        self.assertTrue(model._fused_softmax_ce)
        result = check_gradients(model, X_BINARY, Y_MULTI, tolerance=1e-6, seed=0)
        self.assertTrue(result.passed, result.summary())

    def test_softmax_without_fusion(self):
        # A softmax layer followed by MSE takes the general Jacobian path.
        set_seed(4)
        self._check(Sequential([
            Dense(2, 5, activation="tanh"),
            Dense(5, 3, activation="softmax")]), X_BINARY, Y_MULTI, "mse")

    def test_relu_network(self):
        set_seed(5)
        # Looser tolerance: ReLU's kink limits finite-difference accuracy.
        self._check(Sequential([
            Dense(2, 8, activation="relu"),
            Dense(8, 1, activation="sigmoid")]),
            X_BINARY, Y_BINARY, "binary_cross_entropy", tolerance=1e-5)

    def test_deep_network(self):
        set_seed(6)
        self._check(Sequential([
            Dense(2, 6, activation="tanh"),
            Dense(6, 6, activation="tanh"),
            Dense(6, 4, activation="tanh"),
            Dense(4, 1, activation="sigmoid")]),
            X_BINARY, Y_BINARY, "binary_cross_entropy")

    def test_l2_regularization(self):
        set_seed(7)
        self._check(Sequential([
            Dense(2, 5, activation="tanh", l2=0.05),
            Dense(5, 1, activation="sigmoid", l2=0.05)]),
            X_BINARY, Y_BINARY, "mse")

    def test_l1_regularization(self):
        set_seed(8)
        self._check(Sequential([
            Dense(2, 5, activation="tanh", l1=0.02),
            Dense(5, 1, activation="sigmoid")]),
            X_BINARY, Y_BINARY, "mse", tolerance=1e-5)

    def test_batchnorm(self):
        set_seed(9)
        self._check(Sequential([
            Dense(2, 4, activation="identity"),
            BatchNorm(4),
            Dense(4, 1, activation="sigmoid")]), X_BINARY, Y_BINARY, "mse")

    def test_elu_gelu_swish(self):
        set_seed(10)
        self._check(Sequential([
            Dense(2, 4, activation="elu"),
            Dense(4, 4, activation="gelu"),
            Dense(4, 4, activation="swish"),
            Dense(4, 1, activation="sigmoid")]), X_BINARY, Y_BINARY, "mse")

    def test_leaky_relu(self):
        set_seed(11)
        self._check(Sequential([
            Dense(2, 6, activation="leaky_relu"),
            Dense(6, 1, activation="sigmoid")]),
            X_BINARY, Y_BINARY, "mse", tolerance=1e-5)

    def test_huber_loss(self):
        set_seed(12)
        self._check(Sequential([
            Dense(2, 5, activation="tanh"),
            Dense(5, 1, activation="identity")]), X_BINARY, Y_BINARY, "huber")

    def test_mae_loss(self):
        set_seed(13)
        self._check(Sequential([
            Dense(2, 5, activation="tanh"),
            Dense(5, 1, activation="identity")]),
            X_BINARY, Y_BINARY, "mae", tolerance=1e-5)

    def test_no_bias(self):
        set_seed(14)
        self._check(Sequential([
            Dense(2, 5, activation="tanh", use_bias=False),
            Dense(5, 1, activation="sigmoid", use_bias=False)]),
            X_BINARY, Y_BINARY, "mse")

    def test_residual_block(self):
        set_seed(15)
        self._check(Sequential([
            Dense(2, 5, activation="tanh"),
            Residual([Dense(5, 5, activation="tanh")]),
            Dense(5, 1, activation="sigmoid")]), X_BINARY, Y_BINARY, "mse")

    def test_dropout_is_disabled_during_the_check(self):
        set_seed(16)
        model = Sequential([
            Dense(2, 6, activation="tanh"),
            Dropout(0.5),
            Dense(6, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=SGD(0.01))
        result = check_gradients(model, X_BINARY, Y_BINARY, tolerance=1e-6, seed=0)
        self.assertTrue(result.passed, result.summary())
        # The rate must be restored afterwards.
        self.assertEqual(model.layers[1].rate, 0.5)

    def test_flatten(self):
        set_seed(17)
        X3 = Tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 0.0], [3.0, 2.0]]])
        self._check(Sequential([
            Flatten(),
            Dense(4, 4, activation="tanh"),
            Dense(4, 1, activation="sigmoid")]), X3, [[0.0], [1.0]], "mse")


class TestLayerGradients(unittest.TestCase):
    """Isolated per-layer checks, which localise a failure precisely."""

    def test_dense_alone(self):
        set_seed(20)
        result = check_layer_gradients(
            Dense(3, 4, activation="tanh"),
            Tensor([[0.5, -0.3, 0.8], [1.0, 0.2, -0.5]]))
        self.assertTrue(result.passed, result.summary())

    def test_dense_with_softmax_alone(self):
        set_seed(21)
        result = check_layer_gradients(
            Dense(3, 4, activation="softmax"),
            Tensor([[0.5, -0.3, 0.8], [1.0, 0.2, -0.5]]))
        self.assertTrue(result.passed, result.summary())

    def test_batchnorm_alone(self):
        set_seed(22)
        result = check_layer_gradients(
            BatchNorm(3),
            Tensor([[0.5, -0.3, 0.8], [1.0, 0.2, -0.5], [0.1, 0.9, 0.4]]))
        self.assertTrue(result.passed, result.summary())


class TestCheckerDetectsBrokenGradients(unittest.TestCase):
    """The checker must actually fail when the gradient is wrong.

    A gradient check that passes unconditionally is worse than none at all,
    so this deliberately corrupts a backward pass and asserts detection.
    """

    def test_wrong_gradient_is_caught(self):
        set_seed(30)
        model = Sequential([
            Dense(2, 4, activation="tanh"),
            Dense(4, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=SGD(0.01))

        layer = model.layers[0]
        original_backward = layer.backward

        def corrupted(grad_output, **kwargs):
            result = original_backward(grad_output, **kwargs)
            layer.dW.mul_(1.5)      # a 50% error, the kind a bad transpose makes
            return result

        layer.backward = corrupted
        result = check_gradients(model, X_BINARY, Y_BINARY, tolerance=1e-6, seed=0)
        self.assertFalse(result.passed,
                         "the checker failed to detect a deliberately wrong gradient")
        self.assertGreater(result.max_error, 1e-6)


if __name__ == "__main__":
    unittest.main()
