"""Tests for layers: forward shapes, the backward formulas, and initialization."""

import math
import unittest

from scratch_nn.initialization import (get_initializer, he_normal_init,
                                       xavier_uniform_init, zeros_init)
from scratch_nn.layers import BatchNorm, Dense, Dropout, Flatten, Residual
from scratch_nn.tensor import Tensor, TensorShapeError, zeros
from scratch_nn.utils import set_seed


class TestDenseForward(unittest.TestCase):

    def test_output_shape(self):
        layer = Dense(4, 8)
        out = layer.forward(Tensor([[1.0, 2.0, 3.0, 4.0]]))
        self.assertEqual(out.shape, (1, 8))

    def test_batch_shape(self):
        out = Dense(3, 5).forward(Tensor([[1.0, 2.0, 3.0]] * 7))
        self.assertEqual(out.shape, (7, 5))

    def test_computes_xw_plus_b(self):
        layer = Dense(2, 2, activation="identity")
        layer.W = Tensor([[1.0, 2.0], [3.0, 4.0]])
        layer.b = Tensor([10.0, 20.0])
        out = layer.forward(Tensor([[1.0, 1.0]]))
        # [1,1] @ [[1,2],[3,4]] = [4, 6]; + [10, 20] = [14, 26]
        self.assertEqual(out.tolist(), [[14.0, 26.0]])

    def test_bias_broadcast_across_batch(self):
        layer = Dense(1, 2, activation="identity")
        layer.W = Tensor([[1.0, 1.0]])
        layer.b = Tensor([5.0, -5.0])
        out = layer.forward(Tensor([[1.0], [2.0]]))
        self.assertEqual(out.tolist(), [[6.0, -4.0], [7.0, -3.0]])

    def test_activation_applied(self):
        layer = Dense(2, 2, activation="relu")
        layer.W = Tensor([[1.0, -1.0], [1.0, -1.0]])
        layer.b = zeros((2,))
        out = layer.forward(Tensor([[1.0, 1.0]]))
        self.assertEqual(out.tolist(), [[2.0, 0.0]])   # relu clamps the -2

    def test_no_bias_option(self):
        layer = Dense(2, 2, use_bias=False)
        self.assertIsNone(layer.b)
        self.assertEqual(len(layer.parameters()), 1)

    def test_wrong_feature_count_rejected(self):
        with self.assertRaises(TensorShapeError):
            Dense(4, 2).forward(Tensor([[1.0, 2.0]]))

    def test_requires_2d_input(self):
        with self.assertRaises(TensorShapeError):
            Dense(2, 2).forward(Tensor([1.0, 2.0]))

    def test_invalid_sizes(self):
        with self.assertRaises(ValueError):
            Dense(0, 5)


class TestDenseBackward(unittest.TestCase):

    def _layer(self):
        layer = Dense(2, 3, activation="identity")
        layer.W = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        layer.b = Tensor([0.0, 0.0, 0.0])
        return layer

    def test_dW_equals_x_transpose_times_dz(self):
        layer = self._layer()
        x = Tensor([[1.0, 2.0]])
        layer.forward(x)
        dz = Tensor([[1.0, 1.0, 1.0]])
        layer.backward(dz)
        # dW = x^T @ dz -> column i of dW is x_i * dz
        self.assertEqual(layer.dW.tolist(), [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    def test_db_is_column_sum_of_dz(self):
        layer = self._layer()
        layer.forward(Tensor([[1.0, 2.0], [3.0, 4.0]]))
        layer.backward(Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        self.assertEqual(layer.db.tolist(), [5.0, 7.0, 9.0])

    def test_dX_equals_dz_times_W_transpose(self):
        layer = self._layer()
        layer.forward(Tensor([[1.0, 2.0]]))
        dx = layer.backward(Tensor([[1.0, 1.0, 1.0]]))
        # row sums of W: 1+2+3 = 6, 4+5+6 = 15
        self.assertEqual(dx.tolist(), [[6.0, 15.0]])
        self.assertEqual(dx.shape, (1, 2))

    def test_gradients_accumulate(self):
        layer = self._layer()
        x, dz = Tensor([[1.0, 2.0]]), Tensor([[1.0, 1.0, 1.0]])
        layer.forward(x)
        layer.backward(dz)
        first = list(layer.dW.data)
        layer.forward(x)
        layer.backward(dz)
        for i in range(len(first)):
            self.assertAlmostEqual(layer.dW.data[i], 2 * first[i])

    def test_zero_grad_resets(self):
        layer = self._layer()
        layer.forward(Tensor([[1.0, 2.0]]))
        layer.backward(Tensor([[1.0, 1.0, 1.0]]))
        layer.zero_grad()
        self.assertTrue(all(v == 0.0 for v in layer.dW.data))
        self.assertTrue(all(v == 0.0 for v in layer.db.data))

    def test_backward_before_forward_raises(self):
        with self.assertRaises(RuntimeError):
            Dense(2, 2).backward(Tensor([[1.0, 1.0]]))

    def test_wrong_gradient_shape_rejected(self):
        layer = self._layer()
        layer.forward(Tensor([[1.0, 2.0]]))
        with self.assertRaises(TensorShapeError):
            layer.backward(Tensor([[1.0, 1.0]]))


class TestRegularization(unittest.TestCase):

    def test_l2_penalty_value(self):
        layer = Dense(2, 2, l2=0.1)
        layer.W = Tensor([[1.0, 2.0], [3.0, 4.0]])
        # 0.1 * (1 + 4 + 9 + 16) = 3.0
        self.assertAlmostEqual(layer.regularization_loss(), 3.0)

    def test_l1_penalty_value(self):
        layer = Dense(2, 2, l1=0.5)
        layer.W = Tensor([[1.0, -2.0], [3.0, -4.0]])
        self.assertAlmostEqual(layer.regularization_loss(), 0.5 * 10.0)

    def test_l2_adds_to_weight_gradient(self):
        layer = Dense(1, 1, activation="identity", l2=0.5)
        layer.W = Tensor([[2.0]])
        layer.b = Tensor([0.0])
        layer.forward(Tensor([[0.0]]))       # zero input -> zero data gradient
        layer.backward(Tensor([[0.0]]))
        # only the penalty term remains: d/dW (l2 * W^2) = 2*l2*W = 2.0
        self.assertAlmostEqual(layer.dW.data[0], 2.0)

    def test_bias_is_not_regularized(self):
        layer = Dense(1, 1, activation="identity", l2=0.5)
        layer.b = Tensor([5.0])
        layer.forward(Tensor([[0.0]]))
        layer.backward(Tensor([[0.0]]))
        self.assertAlmostEqual(layer.db.data[0], 0.0)

    def test_no_penalty_when_disabled(self):
        self.assertEqual(Dense(2, 2).regularization_loss(), 0.0)


class TestDropout(unittest.TestCase):

    def test_identity_at_inference(self):
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = Dropout(0.5).forward(x, training=False)
        self.assertEqual(out.tolist(), x.tolist())

    def test_drops_during_training(self):
        set_seed(0)
        out = Dropout(0.5).forward(Tensor([[1.0] * 200]), training=True)
        zeroed = sum(1 for v in out.data if v == 0.0)
        self.assertGreater(zeroed, 50)
        self.assertLess(zeroed, 150)

    def test_inverted_scaling_preserves_expectation(self):
        set_seed(1)
        rate = 0.5
        out = Dropout(rate).forward(Tensor([[1.0] * 4000]), training=True)
        # Surviving units are scaled by 1/(1-rate), so the mean stays near 1.
        self.assertAlmostEqual(sum(out.data) / out.size, 1.0, delta=0.1)

    def test_kept_values_are_scaled(self):
        set_seed(2)
        out = Dropout(0.5).forward(Tensor([[1.0] * 50]), training=True)
        for v in out.data:
            self.assertIn(round(v, 6), (0.0, 2.0))

    def test_backward_uses_same_mask(self):
        set_seed(3)
        layer = Dropout(0.5)
        out = layer.forward(Tensor([[1.0] * 20]), training=True)
        grad = layer.backward(Tensor([[1.0] * 20]))
        for i in range(20):
            if out.data[i] == 0.0:
                self.assertEqual(grad.data[i], 0.0)
            else:
                self.assertAlmostEqual(grad.data[i], 2.0)

    def test_rate_zero_is_identity(self):
        x = Tensor([[1.0, 2.0]])
        self.assertEqual(Dropout(0.0).forward(x, training=True).tolist(), x.tolist())

    def test_invalid_rate(self):
        with self.assertRaises(ValueError):
            Dropout(1.0)
        with self.assertRaises(ValueError):
            Dropout(-0.1)


class TestFlatten(unittest.TestCase):

    def test_collapses_trailing_axes(self):
        out = Flatten().forward(Tensor(list(range(12)), (2, 2, 3)))
        self.assertEqual(out.shape, (2, 6))

    def test_backward_restores_shape(self):
        layer = Flatten()
        layer.forward(Tensor(list(range(12)), (2, 2, 3)))
        self.assertEqual(layer.backward(Tensor([1.0] * 12, (2, 6))).shape, (2, 2, 3))

    def test_values_preserved(self):
        t = Tensor(list(range(12)), (2, 2, 3))
        self.assertEqual(Flatten().forward(t).data, t.data)


class TestBatchNorm(unittest.TestCase):

    def test_normalizes_to_zero_mean_unit_variance(self):
        layer = BatchNorm(2)
        x = Tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        out = layer.forward(x, training=True)
        for c in range(2):
            col = [out.data[r * 2 + c] for r in range(4)]
            mean = sum(col) / 4
            var = sum((v - mean) ** 2 for v in col) / 4
            self.assertAlmostEqual(mean, 0.0, places=6)
            self.assertAlmostEqual(var, 1.0, places=4)

    def test_inference_uses_running_statistics(self):
        layer = BatchNorm(2)
        x = Tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        for _ in range(50):
            layer.forward(x, training=True)
        out = layer.forward(x, training=False)
        self.assertTrue(all(math.isfinite(v) for v in out.data))
        # Running stats should have moved toward the batch statistics.
        self.assertGreater(layer.running_mean.data[1], 5.0)

    def test_single_sample_rejected_in_training(self):
        with self.assertRaises(ValueError):
            BatchNorm(2).forward(Tensor([[1.0, 2.0]]), training=True)

    def test_single_sample_fine_at_inference(self):
        out = BatchNorm(2).forward(Tensor([[1.0, 2.0]]), training=False)
        self.assertEqual(out.shape, (1, 2))

    def test_learnable_parameters(self):
        layer = BatchNorm(3)
        self.assertEqual(len(layer.parameters()), 2)
        self.assertEqual(layer.gamma.tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(layer.beta.tolist(), [0.0, 0.0, 0.0])


class TestResidual(unittest.TestCase):

    def test_adds_input_to_block_output(self):
        block = Dense(3, 3, activation="identity", initialization="zeros")
        block.b = zeros((3,))
        layer = Residual([block])
        x = Tensor([[1.0, 2.0, 3.0]])
        # A zeroed block outputs 0, so the residual output is just x.
        self.assertEqual(layer.forward(x).tolist(), [[1.0, 2.0, 3.0]])

    def test_shape_must_be_preserved(self):
        with self.assertRaises(TensorShapeError):
            Residual([Dense(3, 5)]).forward(Tensor([[1.0, 2.0, 3.0]]))

    def test_gradient_has_identity_path(self):
        block = Dense(2, 2, activation="identity", initialization="zeros")
        layer = Residual([block])
        layer.forward(Tensor([[1.0, 2.0]]))
        grad = layer.backward(Tensor([[1.0, 1.0]]))
        # The block's weights are zero, so all of the gradient arrives via
        # the skip connection unchanged.
        self.assertEqual(grad.tolist(), [[1.0, 1.0]])


class TestInitialization(unittest.TestCase):

    def test_shapes(self):
        self.assertEqual(he_normal_init(4, 8).shape, (4, 8))
        self.assertEqual(xavier_uniform_init(4, 8).shape, (4, 8))

    def test_zeros_init(self):
        self.assertTrue(all(v == 0.0 for v in zeros_init(3, 3).data))

    def test_he_variance_is_two_over_fan_in(self):
        set_seed(0)
        fan_in = 200
        w = he_normal_init(fan_in, 200)
        mean = sum(w.data) / w.size
        var = sum((v - mean) ** 2 for v in w.data) / w.size
        self.assertAlmostEqual(var, 2.0 / fan_in, delta=0.2 * (2.0 / fan_in))

    def test_xavier_uniform_respects_its_limit(self):
        set_seed(0)
        fan_in, fan_out = 50, 50
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        for v in xavier_uniform_init(fan_in, fan_out).data:
            self.assertLessEqual(abs(v), limit + 1e-12)

    def test_symmetry_is_broken_by_random_init(self):
        set_seed(0)
        w = he_normal_init(5, 5)
        self.assertGreater(len(set(w.data)), 20)

    def test_registry(self):
        self.assertIs(get_initializer("he"), he_normal_init)
        self.assertIs(get_initializer("xavier"), xavier_uniform_init)
        with self.assertRaises(ValueError):
            get_initializer("not_an_init")

    def test_dense_accepts_named_initialization(self):
        layer = Dense(4, 8, initialization="he")
        self.assertEqual(layer.W.shape, (4, 8))
        self.assertEqual(layer.initialization, "he")


if __name__ == "__main__":
    unittest.main()
