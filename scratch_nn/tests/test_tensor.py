"""Tests for the tensor engine.

Expected values here are hand-computed, not produced by running the code —
a test that asserts the implementation matches itself proves nothing.
"""

import math
import unittest

from scratch_nn.tensor import (Tensor, TensorShapeError, arange, eye, from_flat,
                               full, ones, rand, randn, stack_rows, zeros)


class TestConstruction(unittest.TestCase):

    def test_shape_inference(self):
        self.assertEqual(Tensor(5).shape, ())
        self.assertEqual(Tensor([1, 2, 3]).shape, (3,))
        self.assertEqual(Tensor([[1, 2], [3, 4]]).shape, (2, 2))
        self.assertEqual(Tensor([[[1], [2]], [[3], [4]]]).shape, (2, 2, 1))

    def test_row_major_flattening(self):
        # (2,3) laid out row by row: strides are (3, 1).
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(t.data, [1, 2, 3, 4, 5, 6])
        self.assertEqual(t.strides, (3, 1))

    def test_tolist_roundtrip(self):
        nested = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        self.assertEqual(Tensor(nested).tolist(), nested)

    def test_ragged_input_rejected(self):
        with self.assertRaises(TensorShapeError):
            Tensor([[1, 2], [3]])
        with self.assertRaises(TensorShapeError):
            Tensor([[1, 2], 3])

    def test_non_numeric_rejected(self):
        with self.assertRaises(TensorShapeError):
            Tensor([[1, "a"]])

    def test_size_and_ndim(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(t.size, 6)
        self.assertEqual(t.ndim, 2)
        self.assertEqual(len(t), 2)

    def test_copy_is_deep(self):
        a = Tensor([[1, 2], [3, 4]])
        b = a.copy()
        b[0, 0] = 99
        self.assertEqual(a[0, 0], 1.0)


class TestFactories(unittest.TestCase):

    def test_zeros_ones_full(self):
        self.assertEqual(zeros((2, 3)).tolist(), [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(ones(3).tolist(), [1, 1, 1])
        self.assertEqual(full((2, 2), 7).tolist(), [[7, 7], [7, 7]])

    def test_eye(self):
        self.assertEqual(eye(3).tolist(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_arange(self):
        self.assertEqual(arange(5).tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(arange(1, 4).tolist(), [1, 2, 3])
        self.assertEqual(arange(0, 1, 0.5).tolist(), [0.0, 0.5])

    def test_random_shapes_and_bounds(self):
        self.assertEqual(rand((3, 4)).shape, (3, 4))
        self.assertEqual(randn((2, 2)).shape, (2, 2))
        for v in rand((50,), low=-1.0, high=1.0).data:
            self.assertTrue(-1.0 <= v < 1.0)

    def test_stack_rows_rejects_uneven(self):
        self.assertEqual(stack_rows([[1, 2], [3, 4]]).shape, (2, 2))
        with self.assertRaises(TensorShapeError):
            stack_rows([[1, 2], [3]])

    def test_from_flat(self):
        self.assertEqual(from_flat([1, 2, 3, 4], (2, 2)).tolist(), [[1, 2], [3, 4]])


class TestIndexing(unittest.TestCase):

    def test_element_access(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(t[0, 0], 1.0)
        self.assertEqual(t[1, 2], 6.0)
        self.assertEqual(t[-1, -1], 6.0)

    def test_row_access_returns_tensor(self):
        row = Tensor([[1, 2, 3], [4, 5, 6]])[1]
        self.assertIsInstance(row, Tensor)
        self.assertEqual(row.tolist(), [4, 5, 6])

    def test_assignment(self):
        t = zeros((2, 2))
        t[0, 1] = 5
        self.assertEqual(t.tolist(), [[0, 5], [0, 0]])

    def test_out_of_bounds(self):
        t = Tensor([[1, 2], [3, 4]])
        with self.assertRaises(IndexError):
            _ = t[5, 0]
        with self.assertRaises(TensorShapeError):
            _ = t[0, 0, 0]

    def test_iteration_over_first_axis(self):
        rows = [r.tolist() for r in Tensor([[1, 2], [3, 4]])]
        self.assertEqual(rows, [[1, 2], [3, 4]])

    def test_item(self):
        self.assertEqual(Tensor([[42]]).item(), 42.0)
        with self.assertRaises(TensorShapeError):
            Tensor([1, 2]).item()


class TestArithmetic(unittest.TestCase):

    def test_addition(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        self.assertEqual(a.add(b).tolist(), [[6, 8], [10, 12]])
        self.assertEqual((a + b).tolist(), [[6, 8], [10, 12]])

    def test_subtraction(self):
        a = Tensor([[5, 6], [7, 8]])
        b = Tensor([[1, 2], [3, 4]])
        self.assertEqual((a - b).tolist(), [[4, 4], [4, 4]])

    def test_elementwise_multiplication(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        # Hadamard product, NOT a matrix product.
        self.assertEqual((a * b).tolist(), [[5, 12], [21, 32]])

    def test_scalar_operations(self):
        a = Tensor([[1, 2], [3, 4]])
        self.assertEqual((a * 2).tolist(), [[2, 4], [6, 8]])
        self.assertEqual((2 * a).tolist(), [[2, 4], [6, 8]])
        self.assertEqual((a + 10).tolist(), [[11, 12], [13, 14]])
        self.assertEqual((a / 2).tolist(), [[0.5, 1.0], [1.5, 2.0]])

    def test_division_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            Tensor([1.0]).div(Tensor([0.0]))

    def test_power_negation_abs(self):
        a = Tensor([[1, -2], [3, -4]])
        self.assertEqual(a.pow(2).tolist(), [[1, 4], [9, 16]])
        self.assertEqual((-a).tolist(), [[-1, 2], [-3, 4]])
        self.assertEqual(abs(a).tolist(), [[1, 2], [3, 4]])

    def test_exp_saturates_instead_of_raising(self):
        # math.exp(1000) would raise OverflowError.
        self.assertEqual(Tensor([1000.0]).exp().data[0], math.inf)
        self.assertEqual(Tensor([-1000.0]).exp().data[0], 0.0)

    def test_log_guards_zero(self):
        with self.assertRaises(ValueError):
            Tensor([0.0]).log()
        self.assertTrue(math.isfinite(Tensor([0.0]).log(eps=1e-12).data[0]))

    def test_clip(self):
        self.assertEqual(Tensor([-5, 0, 5]).clip(-1, 1).tolist(), [-1, 0, 1])

    def test_in_place_helpers(self):
        t = Tensor([1.0, 2.0])
        t.add_(Tensor([1.0, 1.0]))
        self.assertEqual(t.tolist(), [2.0, 3.0])
        t.mul_(2)
        self.assertEqual(t.tolist(), [4.0, 6.0])
        t.zero_()
        self.assertEqual(t.tolist(), [0.0, 0.0])


class TestBroadcasting(unittest.TestCase):

    def test_matrix_plus_row_vector(self):
        # (2,3) + (3,) -> the vector is added to every row.
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(a.add(Tensor([10, 20, 30])).tolist(),
                         [[11, 22, 33], [14, 25, 36]])

    def test_column_times_row(self):
        # (2,1) * (1,3) -> (2,3), the outer product pattern.
        a = Tensor([[1], [2]])
        b = Tensor([[10, 20, 30]])
        self.assertEqual(a.mul(b).tolist(), [[10, 20, 30], [20, 40, 60]])

    def test_incompatible_shapes_rejected(self):
        with self.assertRaises(TensorShapeError):
            Tensor([[1, 2], [3, 4]]).add(Tensor([[1, 2, 3], [4, 5, 6]]))


class TestMatmul(unittest.TestCase):

    def test_known_product(self):
        # [[1,2],[3,4]] @ [[5,6],[7,8]]
        #   row0: 1*5+2*7=19, 1*6+2*8=22
        #   row1: 3*5+4*7=43, 3*6+4*8=50
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        self.assertEqual(a.matmul(b).tolist(), [[19, 22], [43, 50]])
        self.assertEqual((a @ b).tolist(), [[19, 22], [43, 50]])

    def test_non_square(self):
        a = Tensor([[1, 2, 3]])           # (1,3)
        b = Tensor([[1], [2], [3]])       # (3,1)
        self.assertEqual(a.matmul(b).tolist(), [[14]])   # 1+4+9
        self.assertEqual(b.matmul(a).tolist(),
                         [[1, 2, 3], [2, 4, 6], [3, 6, 9]])

    def test_identity_is_neutral(self):
        a = Tensor([[1, 2], [3, 4]])
        self.assertEqual(a.matmul(eye(2)).tolist(), a.tolist())

    def test_inner_dimension_must_agree(self):
        with self.assertRaises(TensorShapeError):
            Tensor([[1, 2], [3, 4]]).matmul(Tensor([[1, 2, 3]]))

    def test_requires_2d(self):
        with self.assertRaises(TensorShapeError):
            Tensor([1, 2, 3]).matmul(Tensor([1, 2, 3]))

    def test_associativity(self):
        a, b, c = Tensor([[1, 2], [3, 4]]), Tensor([[5, 6], [7, 8]]), Tensor([[1, 0], [0, 2]])
        left = a.matmul(b).matmul(c).tolist()
        right = a.matmul(b.matmul(c)).tolist()
        self.assertEqual(left, right)


class TestTranspose(unittest.TestCase):

    def test_2d(self):
        self.assertEqual(Tensor([[1, 2, 3], [4, 5, 6]]).T.tolist(),
                         [[1, 4], [2, 5], [3, 6]])

    def test_double_transpose_is_identity(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(a.T.T.tolist(), a.tolist())

    def test_transpose_of_product(self):
        # (AB)^T == B^T A^T
        a, b = Tensor([[1, 2], [3, 4]]), Tensor([[5, 6], [7, 8]])
        self.assertEqual(a.matmul(b).T.tolist(), b.T.matmul(a.T).tolist())

    def test_3d_permutation(self):
        t = Tensor(list(range(24)), (2, 3, 4))
        p = t.transpose(2, 0, 1)
        self.assertEqual(p.shape, (4, 2, 3))
        # Element (i,j,k) must land at (k,i,j).
        self.assertEqual(p[2, 1, 0], t[1, 0, 2])

    def test_invalid_permutation(self):
        with self.assertRaises(TensorShapeError):
            Tensor([[1, 2], [3, 4]]).transpose(0, 0)


class TestReshape(unittest.TestCase):

    def test_reshape(self):
        self.assertEqual(arange(6).reshape(2, 3).tolist(), [[0, 1, 2], [3, 4, 5]])

    def test_inferred_dimension(self):
        self.assertEqual(arange(12).reshape(3, -1).shape, (3, 4))
        self.assertEqual(arange(12).reshape(-1, 2).shape, (6, 2))

    def test_element_count_must_match(self):
        with self.assertRaises(TensorShapeError):
            arange(6).reshape(4, 2)

    def test_only_one_inferred_dimension(self):
        with self.assertRaises(TensorShapeError):
            arange(12).reshape(-1, -1)

    def test_flatten(self):
        self.assertEqual(Tensor([[1, 2], [3, 4]]).flatten().tolist(), [1, 2, 3, 4])


class TestReductions(unittest.TestCase):

    def setUp(self):
        self.t = Tensor([[1, 2], [3, 4]])

    def test_global_reductions(self):
        self.assertEqual(self.t.sum(), 10.0)
        self.assertEqual(self.t.mean(), 2.5)
        self.assertEqual(self.t.max(), 4.0)
        self.assertEqual(self.t.min(), 1.0)

    def test_axis_reductions(self):
        # axis=0 collapses rows (column sums); axis=1 collapses columns.
        self.assertEqual(self.t.sum(axis=0).tolist(), [4, 6])
        self.assertEqual(self.t.sum(axis=1).tolist(), [3, 7])
        self.assertEqual(self.t.mean(axis=0).tolist(), [2, 3])
        self.assertEqual(self.t.max(axis=1).tolist(), [2, 4])
        self.assertEqual(self.t.min(axis=0).tolist(), [1, 2])

    def test_keepdims(self):
        self.assertEqual(self.t.sum(axis=1, keepdims=True).shape, (2, 1))

    def test_3d_axis_reduction_matches_manual(self):
        t = Tensor(list(range(24)), (2, 3, 4))
        nested = t.tolist()
        expected = [[sum(nested[i][j][k] for i in range(2)) for k in range(4)]
                    for j in range(3)]
        self.assertEqual(t.sum(axis=0).tolist(), expected)

    def test_argmax(self):
        self.assertEqual(Tensor([[1, 9, 2], [7, 3, 4]]).argmax(axis=1).tolist(),
                         [1, 0])
        self.assertEqual(Tensor([1, 5, 3]).argmax(), 1)

    def test_bad_axis(self):
        with self.assertRaises(TensorShapeError):
            self.t.sum(axis=5)


class TestLinearAlgebraHelpers(unittest.TestCase):

    def test_dot(self):
        self.assertEqual(Tensor([1, 2, 3]).dot(Tensor([4, 5, 6])), 32.0)

    def test_dot_requires_matching_1d(self):
        with self.assertRaises(TensorShapeError):
            Tensor([1, 2]).dot(Tensor([1, 2, 3]))

    def test_norm(self):
        self.assertAlmostEqual(Tensor([3, 4]).norm(), 5.0)

    def test_is_finite(self):
        self.assertTrue(Tensor([1, 2, 3]).is_finite())
        self.assertFalse(Tensor([1, float("inf")]).is_finite())
        self.assertFalse(Tensor([1, float("nan")]).is_finite())

    def test_equal_with_tolerance(self):
        self.assertTrue(Tensor([1.0]).equal(Tensor([1.0 + 1e-12])))
        self.assertFalse(Tensor([1.0]).equal(Tensor([1.1])))


if __name__ == "__main__":
    unittest.main()
