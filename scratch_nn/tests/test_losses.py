"""Tests for loss functions, against hand-computed values."""

import math
import unittest

from scratch_nn.losses import (BinaryCrossEntropy, CategoricalCrossEntropy,
                               Huber, MeanAbsoluteError, MeanSquaredError,
                               get_loss)
from scratch_nn.tensor import Tensor, TensorShapeError


class TestMeanSquaredError(unittest.TestCase):

    def test_perfect_prediction_is_zero(self):
        loss = MeanSquaredError()
        y = Tensor([[1.0], [2.0]])
        self.assertEqual(loss.forward(y, y), 0.0)

    def test_known_value(self):
        # errors 1 and 1 -> (1 + 1)/2 = 1.0
        loss = MeanSquaredError()
        self.assertAlmostEqual(
            loss.forward(Tensor([[3.0], [5.0]]), Tensor([[2.0], [4.0]])), 1.0)

    def test_gradient_formula(self):
        # dL/dp = 2(p - t)/n, with n = 2 elements
        loss = MeanSquaredError()
        grad = loss.backward(Tensor([[3.0], [5.0]]), Tensor([[2.0], [4.0]]))
        self.assertAlmostEqual(grad.data[0], 2 * 1.0 / 2)
        self.assertAlmostEqual(grad.data[1], 2 * 1.0 / 2)

    def test_gradient_matches_finite_difference(self):
        loss = MeanSquaredError()
        pred = Tensor([[0.3], [0.8], [-0.2]])
        true = Tensor([[0.0], [1.0], [0.5]])
        analytic = loss.backward(pred, true)
        h = 1e-6
        for i in range(pred.size):
            original = pred.data[i]
            pred.data[i] = original + h
            up = loss.forward(pred, true)
            pred.data[i] = original - h
            down = loss.forward(pred, true)
            pred.data[i] = original
            self.assertAlmostEqual(analytic.data[i], (up - down) / (2 * h), places=6)

    def test_shape_mismatch(self):
        with self.assertRaises(TensorShapeError):
            MeanSquaredError().forward(Tensor([[1.0]]), Tensor([[1.0], [2.0]]))


class TestMeanAbsoluteError(unittest.TestCase):

    def test_known_value(self):
        loss = MeanAbsoluteError()
        self.assertAlmostEqual(
            loss.forward(Tensor([[3.0], [1.0]]), Tensor([[2.0], [4.0]])),
            (1.0 + 3.0) / 2)

    def test_gradient_is_sign_over_n(self):
        loss = MeanAbsoluteError()
        grad = loss.backward(Tensor([[3.0], [1.0]]), Tensor([[2.0], [4.0]]))
        self.assertAlmostEqual(grad.data[0], 0.5)    # +1/2
        self.assertAlmostEqual(grad.data[1], -0.5)   # -1/2

    def test_gradient_at_zero_error(self):
        loss = MeanAbsoluteError()
        grad = loss.backward(Tensor([[1.0]]), Tensor([[1.0]]))
        self.assertEqual(grad.data[0], 0.0)


class TestBinaryCrossEntropy(unittest.TestCase):

    def test_confident_and_correct_is_near_zero(self):
        loss = BinaryCrossEntropy()
        value = loss.forward(Tensor([[0.999], [0.001]]), Tensor([[1.0], [0.0]]))
        self.assertLess(value, 0.002)

    def test_maximum_uncertainty(self):
        # p = 0.5 gives -log(0.5) = ln 2 for either label.
        loss = BinaryCrossEntropy()
        value = loss.forward(Tensor([[0.5], [0.5]]), Tensor([[1.0], [0.0]]))
        self.assertAlmostEqual(value, math.log(2), places=10)

    def test_known_value(self):
        # y=1, p=0.8 -> -log(0.8)
        loss = BinaryCrossEntropy()
        self.assertAlmostEqual(loss.forward(Tensor([[0.8]]), Tensor([[1.0]])),
                               -math.log(0.8), places=10)

    def test_confident_and_wrong_is_large_but_finite(self):
        # log(0) would be -inf; clipping keeps it finite.
        loss = BinaryCrossEntropy()
        value = loss.forward(Tensor([[0.0]]), Tensor([[1.0]]))
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 10.0)

    def test_gradient_matches_finite_difference(self):
        loss = BinaryCrossEntropy()
        pred = Tensor([[0.3], [0.7], [0.5]])
        true = Tensor([[0.0], [1.0], [1.0]])
        analytic = loss.backward(pred, true)
        h = 1e-7
        for i in range(pred.size):
            original = pred.data[i]
            pred.data[i] = original + h
            up = loss.forward(pred, true)
            pred.data[i] = original - h
            down = loss.forward(pred, true)
            pred.data[i] = original
            self.assertAlmostEqual(analytic.data[i], (up - down) / (2 * h), places=4)

    def test_labels_outside_unit_interval_rejected(self):
        with self.assertRaises(ValueError):
            BinaryCrossEntropy().forward(Tensor([[0.5]]), Tensor([[2.0]]))


class TestCategoricalCrossEntropy(unittest.TestCase):

    def test_perfect_prediction(self):
        loss = CategoricalCrossEntropy()
        value = loss.forward(Tensor([[1.0, 0.0, 0.0]]), Tensor([[1.0, 0.0, 0.0]]))
        self.assertAlmostEqual(value, 0.0, places=10)

    def test_known_value(self):
        # true class has p = 0.7 -> -log(0.7)
        loss = CategoricalCrossEntropy()
        value = loss.forward(Tensor([[0.7, 0.2, 0.1]]), Tensor([[1.0, 0.0, 0.0]]))
        self.assertAlmostEqual(value, -math.log(0.7), places=10)

    def test_uniform_prediction_over_c_classes(self):
        # -log(1/C) = log(C)
        loss = CategoricalCrossEntropy()
        value = loss.forward(Tensor([[0.25] * 4]), Tensor([[1.0, 0.0, 0.0, 0.0]]))
        self.assertAlmostEqual(value, math.log(4), places=10)

    def test_averages_over_batch(self):
        loss = CategoricalCrossEntropy()
        pred = Tensor([[0.7, 0.3], [0.4, 0.6]])
        true = Tensor([[1.0, 0.0], [0.0, 1.0]])
        expected = (-math.log(0.7) - math.log(0.6)) / 2
        self.assertAlmostEqual(loss.forward(pred, true), expected, places=10)

    def test_zero_probability_is_finite(self):
        loss = CategoricalCrossEntropy()
        value = loss.forward(Tensor([[0.0, 1.0]]), Tensor([[1.0, 0.0]]))
        self.assertTrue(math.isfinite(value))

    def test_fused_gradient_is_p_minus_y(self):
        loss = CategoricalCrossEntropy()
        p = Tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        y = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        grad = loss.backward_fused(p, y)
        for i in range(6):
            self.assertAlmostEqual(grad.data[i], (p.data[i] - y.data[i]) / 2,
                                   places=12)

    def test_fused_gradient_rows_sum_to_zero(self):
        # Softmax outputs and one-hot targets both sum to 1 per row, so their
        # difference must sum to 0.
        loss = CategoricalCrossEntropy()
        grad = loss.backward_fused(Tensor([[0.5, 0.3, 0.2]]),
                                   Tensor([[1.0, 0.0, 0.0]]))
        self.assertAlmostEqual(sum(grad.data), 0.0, places=12)

    def test_generic_gradient_matches_finite_difference(self):
        loss = CategoricalCrossEntropy()
        pred = Tensor([[0.6, 0.3, 0.1]])
        true = Tensor([[1.0, 0.0, 0.0]])
        analytic = loss.backward(pred, true)
        h = 1e-7
        for i in range(pred.size):
            original = pred.data[i]
            pred.data[i] = original + h
            up = loss.forward(pred, true)
            pred.data[i] = original - h
            down = loss.forward(pred, true)
            pred.data[i] = original
            self.assertAlmostEqual(analytic.data[i], (up - down) / (2 * h), places=4)


class TestHuber(unittest.TestCase):

    def test_quadratic_region(self):
        # |error| = 0.5 <= delta=1, so L = 0.5 * 0.25
        loss = Huber(delta=1.0)
        self.assertAlmostEqual(
            loss.forward(Tensor([[0.5]]), Tensor([[0.0]])), 0.125)

    def test_linear_region(self):
        # |error| = 5 > delta=1, so L = 1*(5 - 0.5) = 4.5
        loss = Huber(delta=1.0)
        self.assertAlmostEqual(
            loss.forward(Tensor([[5.0]]), Tensor([[0.0]])), 4.5)

    def test_continuous_at_the_boundary(self):
        loss = Huber(delta=1.0)
        just_below = loss.forward(Tensor([[0.999999]]), Tensor([[0.0]]))
        just_above = loss.forward(Tensor([[1.000001]]), Tensor([[0.0]]))
        self.assertAlmostEqual(just_below, just_above, places=5)

    def test_gradient_is_bounded_by_delta(self):
        loss = Huber(delta=1.0)
        grad = loss.backward(Tensor([[100.0]]), Tensor([[0.0]]))
        self.assertAlmostEqual(grad.data[0], 1.0)   # capped, not 100

    def test_delta_must_be_positive(self):
        with self.assertRaises(ValueError):
            Huber(delta=0.0)


class TestRegistry(unittest.TestCase):

    def test_lookup_by_name(self):
        self.assertIsInstance(get_loss("mse"), MeanSquaredError)
        self.assertIsInstance(get_loss("binary_cross_entropy"), BinaryCrossEntropy)
        self.assertIsInstance(get_loss("categorical_cross_entropy"),
                              CategoricalCrossEntropy)

    def test_unknown_name(self):
        with self.assertRaises(ValueError):
            get_loss("not_a_loss")

    def test_empty_batch_rejected(self):
        with self.assertRaises(TensorShapeError):
            MeanSquaredError().forward(Tensor([], (0,)), Tensor([], (0,)))


if __name__ == "__main__":
    unittest.main()
