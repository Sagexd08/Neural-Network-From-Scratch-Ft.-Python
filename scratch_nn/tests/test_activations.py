"""Tests for activation functions.

Outputs are checked against values computed by hand from the defining
formula, and derivatives against the analytic expression evaluated
independently.
"""

import math
import unittest

from scratch_nn.activations import (ELU, GELU, Identity, LeakyReLU, ReLU,
                                    Sigmoid, Softmax, Swish, Tanh,
                                    get_activation)
from scratch_nn.tensor import Tensor


class TestReLU(unittest.TestCase):

    def test_forward(self):
        out = ReLU().forward(Tensor([[-2.0, -0.5, 0.0, 0.5, 2.0]]))
        self.assertEqual(out.tolist(), [[0.0, 0.0, 0.0, 0.5, 2.0]])

    def test_backward_gates_by_sign(self):
        relu = ReLU()
        relu.forward(Tensor([[-2.0, -0.5, 0.0, 0.5, 2.0]]))
        grad = relu.backward(Tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]))
        # f'(x) = 1 for x>0, else 0 (subgradient 0 at the kink).
        self.assertEqual(grad.tolist(), [[0.0, 0.0, 0.0, 1.0, 1.0]])

    def test_backward_scales_incoming_gradient(self):
        relu = ReLU()
        relu.forward(Tensor([[3.0, -3.0]]))
        self.assertEqual(relu.backward(Tensor([[7.0, 7.0]])).tolist(), [[7.0, 0.0]])

    def test_backward_before_forward_raises(self):
        with self.assertRaises(RuntimeError):
            ReLU().backward(Tensor([[1.0]]))


class TestLeakyReLU(unittest.TestCase):

    def test_forward_has_negative_slope(self):
        out = LeakyReLU(alpha=0.1).forward(Tensor([[-2.0, 3.0]]))
        self.assertAlmostEqual(out.data[0], -0.2)
        self.assertAlmostEqual(out.data[1], 3.0)

    def test_backward(self):
        act = LeakyReLU(alpha=0.1)
        act.forward(Tensor([[-2.0, 3.0]]))
        grad = act.backward(Tensor([[1.0, 1.0]]))
        self.assertAlmostEqual(grad.data[0], 0.1)
        self.assertAlmostEqual(grad.data[1], 1.0)

    def test_negative_alpha_rejected(self):
        with self.assertRaises(ValueError):
            LeakyReLU(alpha=-0.5)


class TestSigmoid(unittest.TestCase):

    def test_known_values(self):
        out = Sigmoid().forward(Tensor([[0.0, 1.0, -1.0]]))
        self.assertAlmostEqual(out.data[0], 0.5)
        self.assertAlmostEqual(out.data[1], 1 / (1 + math.exp(-1)), places=12)
        self.assertAlmostEqual(out.data[2], 1 / (1 + math.exp(1)), places=12)

    def test_output_bounded(self):
        out = Sigmoid().forward(Tensor([[-100.0, 100.0]]))
        self.assertTrue(0.0 <= out.data[0] < 1e-9)
        self.assertTrue(1.0 - 1e-9 < out.data[1] <= 1.0)

    def test_no_overflow_on_extremes(self):
        # The naive 1/(1+exp(-x)) would raise OverflowError at -800.
        out = Sigmoid().forward(Tensor([[-800.0, 800.0]]))
        self.assertTrue(all(math.isfinite(v) for v in out.data))

    def test_derivative_is_s_times_one_minus_s(self):
        act = Sigmoid()
        out = act.forward(Tensor([[0.0, 2.0, -2.0]]))
        grad = act.backward(Tensor([[1.0, 1.0, 1.0]]))
        for i, s in enumerate(out.data):
            self.assertAlmostEqual(grad.data[i], s * (1 - s), places=12)
        # At x=0, s=0.5 so the derivative is exactly 0.25 (its maximum).
        self.assertAlmostEqual(grad.data[0], 0.25)


class TestTanh(unittest.TestCase):

    def test_known_values(self):
        out = Tanh().forward(Tensor([[0.0, 1.0, -1.0]]))
        self.assertAlmostEqual(out.data[0], 0.0)
        self.assertAlmostEqual(out.data[1], math.tanh(1.0), places=12)
        self.assertAlmostEqual(out.data[2], -math.tanh(1.0), places=12)

    def test_zero_centred_and_odd(self):
        out = Tanh().forward(Tensor([[2.0, -2.0]]))
        self.assertAlmostEqual(out.data[0], -out.data[1])

    def test_derivative(self):
        act = Tanh()
        out = act.forward(Tensor([[0.0, 1.5]]))
        grad = act.backward(Tensor([[1.0, 1.0]]))
        for i, t in enumerate(out.data):
            self.assertAlmostEqual(grad.data[i], 1 - t * t, places=12)
        self.assertAlmostEqual(grad.data[0], 1.0)   # tanh'(0) == 1


class TestSoftmax(unittest.TestCase):

    def test_rows_sum_to_one(self):
        out = Softmax().forward(Tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
        self.assertAlmostEqual(sum(out.data[0:3]), 1.0, places=12)
        self.assertAlmostEqual(sum(out.data[3:6]), 1.0, places=12)

    def test_uniform_input_gives_uniform_output(self):
        out = Softmax().forward(Tensor([[5.0, 5.0, 5.0, 5.0]]))
        for v in out.data:
            self.assertAlmostEqual(v, 0.25, places=12)

    def test_known_values(self):
        out = Softmax().forward(Tensor([[1.0, 2.0, 3.0]]))
        exps = [math.exp(v - 3.0) for v in (1.0, 2.0, 3.0)]
        total = sum(exps)
        for i in range(3):
            self.assertAlmostEqual(out.data[i], exps[i] / total, places=12)

    def test_large_logits_do_not_overflow(self):
        # Without the max-subtraction trick exp(1000) is inf and inf/inf is nan.
        out = Softmax().forward(Tensor([[1000.0, 1001.0, 1002.0]]))
        self.assertTrue(all(math.isfinite(v) for v in out.data))
        self.assertAlmostEqual(sum(out.data), 1.0, places=12)

    def test_shift_invariance(self):
        a = Softmax().forward(Tensor([[1.0, 2.0, 3.0]]))
        b = Softmax().forward(Tensor([[101.0, 102.0, 103.0]]))
        for i in range(3):
            self.assertAlmostEqual(a.data[i], b.data[i], places=12)

    def test_requires_2d(self):
        with self.assertRaises(ValueError):
            Softmax().forward(Tensor([1.0, 2.0]))

    def test_backward_against_explicit_jacobian(self):
        act = Softmax()
        p = act.forward(Tensor([[1.0, 2.0, 3.0]]))
        upstream = Tensor([[0.5, -1.0, 2.0]])
        grad = act.backward(upstream)
        # Build the full Jacobian J[i][j] = p_i(delta_ij - p_j) and apply it.
        pv = p.data
        for j in range(3):
            expected = sum(
                upstream.data[i] * pv[i] * ((1.0 if i == j else 0.0) - pv[j])
                for i in range(3))
            self.assertAlmostEqual(grad.data[j], expected, places=12)

    def test_backward_of_constant_gradient_is_zero(self):
        # Softmax outputs are constrained to sum to 1, so pushing all outputs
        # up equally has no effect on the inputs.
        act = Softmax()
        act.forward(Tensor([[1.0, 2.0, 3.0]]))
        grad = act.backward(Tensor([[1.0, 1.0, 1.0]]))
        for v in grad.data:
            self.assertAlmostEqual(v, 0.0, places=12)


class TestSmoothActivations(unittest.TestCase):

    def _numeric_derivative(self, act_factory, x, h=1e-6):
        plus = act_factory().forward(Tensor([[x + h]])).data[0]
        minus = act_factory().forward(Tensor([[x - h]])).data[0]
        return (plus - minus) / (2 * h)

    def test_elu(self):
        out = ELU(alpha=1.0).forward(Tensor([[-1.0, 2.0]]))
        self.assertAlmostEqual(out.data[0], math.exp(-1.0) - 1.0, places=12)
        self.assertAlmostEqual(out.data[1], 2.0)

    def test_elu_derivative_matches_numeric(self):
        for x in (-2.0, -0.3, 0.7, 3.0):
            act = ELU()
            act.forward(Tensor([[x]]))
            analytic = act.backward(Tensor([[1.0]])).data[0]
            self.assertAlmostEqual(analytic, self._numeric_derivative(ELU, x),
                                   places=5)

    def test_gelu_derivative_matches_numeric(self):
        for x in (-2.0, -0.5, 0.0, 1.0, 2.5):
            act = GELU()
            act.forward(Tensor([[x]]))
            analytic = act.backward(Tensor([[1.0]])).data[0]
            self.assertAlmostEqual(analytic, self._numeric_derivative(GELU, x),
                                   places=5)

    def test_swish_derivative_matches_numeric(self):
        for x in (-2.0, -0.5, 0.0, 1.0, 2.5):
            act = Swish()
            act.forward(Tensor([[x]]))
            analytic = act.backward(Tensor([[1.0]])).data[0]
            self.assertAlmostEqual(analytic, self._numeric_derivative(Swish, x),
                                   places=5)

    def test_swish_at_zero(self):
        self.assertAlmostEqual(Swish().forward(Tensor([[0.0]])).data[0], 0.0)


class TestIdentity(unittest.TestCase):

    def test_passthrough(self):
        act = Identity()
        x = Tensor([[1.0, -2.0]])
        self.assertEqual(act.forward(x).tolist(), x.tolist())
        self.assertEqual(act.backward(Tensor([[3.0, 4.0]])).tolist(), [[3.0, 4.0]])


class TestRegistry(unittest.TestCase):

    def test_lookup_by_name(self):
        self.assertIsInstance(get_activation("relu"), ReLU)
        self.assertIsInstance(get_activation("SIGMOID"), Sigmoid)
        self.assertIsInstance(get_activation("softmax"), Softmax)
        self.assertIsInstance(get_activation(None), Identity)

    def test_instance_passthrough(self):
        act = ReLU()
        self.assertIs(get_activation(act), act)

    def test_fresh_instance_per_lookup(self):
        # Activations cache per-batch state, so they must not be shared.
        self.assertIsNot(get_activation("relu"), get_activation("relu"))

    def test_unknown_name(self):
        with self.assertRaises(ValueError):
            get_activation("not_an_activation")

    def test_kwargs_forwarded(self):
        self.assertEqual(get_activation("leaky_relu", alpha=0.2).alpha, 0.2)


if __name__ == "__main__":
    unittest.main()
