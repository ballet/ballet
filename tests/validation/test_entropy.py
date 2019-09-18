import unittest

import numpy as np

from ballet.util import asarray2d
from ballet.validation.entropy import (
    _compute_epsilon, _estimate_cont_entropy, _estimate_disc_entropy,
    _is_column_cont, _is_column_disc, estimate_conditional_information,
    estimate_entropy, estimate_mutual_information)


class EntropyTest(unittest.TestCase):

    def test_disc_entropy_constant_vals_1d(self):
        """If x (column vector) is constant, then H(x) = 0"""
        same_val_arr_ones = np.ones((50, 1))
        H_hat = _estimate_disc_entropy(same_val_arr_ones)
        self.assertEqual(0, H_hat)

    def test_disc_entropy_constant_vals_2d(self):
        """If each column in x (matrix), then H(x) = 0"""
        same_val_arr_zero = np.zeros((50, 1))
        same_val_arr_ones = np.ones((50, 1))
        same_multi_val_arr = np.concatenate(
            (same_val_arr_ones, same_val_arr_zero), axis=1)
        H_hat = _estimate_disc_entropy(same_multi_val_arr)
        self.assertEqual(0, H_hat)

    def test_disc_entropy_two_values(self):
        """Entropy of fair coin ~= log(2)"""
        same_val_arr_zero = np.zeros((50, 1))
        same_val_arr_ones = np.ones((50, 1))
        diff_val_arr = np.concatenate(
            (same_val_arr_ones, same_val_arr_zero), axis=0)

        expected_h = np.log(2)
        H_hat = _estimate_disc_entropy(diff_val_arr)
        self.assertAlmostEqual(expected_h, H_hat)

    def test_is_column_disc(self):
        x = asarray2d(np.arange(50))
        result = _is_column_disc(x)
        self.assertTrue(result)

    def test_is_column_cont(self):
        x = asarray2d(np.random.rand(50))
        result = _is_column_cont(x)
        self.assertTrue(result)

    def test_cont_disc_entropy_differs_disc(self):
        """Expect cont, disc columns to have different entropy"""
        disc = asarray2d(np.arange(50))
        epsilon = _compute_epsilon(disc)

        self.assertNotEqual(
            _estimate_cont_entropy(disc, epsilon), _estimate_disc_entropy(
                disc))

    def test_cont_disc_entropy_differs_disc(self):
        """Expect cont, disc columns to have different entropy"""
        cont = asarray2d(np.arange(50)) + 0.5
        epsilon = _compute_epsilon(cont)
        self.assertNotEqual(
            _estimate_cont_entropy(cont, epsilon), _estimate_disc_entropy(
                cont))

    def test_entropy_multiple_disc(self):
        same_val_arr_zero = np.zeros((50, 1))
        same_val_arr_ones = np.ones((50, 1))
        # The 0.5 forces float => classified as continuous
        cont_val_arange = asarray2d(np.arange(50) + 0.5)
        all_disc_arr = np.concatenate(
            (same_val_arr_ones, same_val_arr_zero), axis=1)
        mixed_val_arr = np.concatenate(
            (all_disc_arr, cont_val_arange), axis=1)

        all_disc_h = estimate_entropy(all_disc_arr)
        mixed_h = estimate_entropy(mixed_val_arr)
        self.assertGreater(
            mixed_h,
            all_disc_h,
            msg='Expected adding continuous column increases entropy')

    def test_mi_uninformative(self):
        x = np.reshape(np.arange(1, 101), (-1, 1))
        y = np.ones((100, 1))
        mi = estimate_mutual_information(x, y)
        h_z = estimate_entropy(x)
        self.assertGreater(
            h_z / 4,
            mi,
            'uninformative column should have no information')

    def test_mi_informative(self):
        x = np.reshape(np.arange(1, 101), (-1, 1))
        y = np.reshape(np.arange(1, 101), (-1, 1))
        mi = estimate_mutual_information(x, y)
        h_y = estimate_entropy(y)
        self.assertGreater(
            mi,
            h_y / 4,
            'exact copy columns should have high information')

    def test_cmi_high_info_uninformative_z(self):
        # redundant copies have little information
        x = np.reshape(np.arange(1, 101), (-1, 1))
        y = np.reshape(np.arange(1, 101), (-1, 1))

        # exact copies of y should have lots of information
        useless_z = np.ones((100, 1))
        cmi = estimate_conditional_information(x, y, useless_z)
        mi = estimate_mutual_information(x, y)
        self.assertAlmostEqual(
            cmi,
            mi,
            'uninformative z should not affect mutual information score')

    def test_cmi_redundant_info(self):
        x = np.reshape(np.arange(1, 101), (-1, 1))
        y = np.reshape(np.arange(1, 101), (-1, 1))
        exact_z = np.reshape(np.arange(1, 101), (-1, 1))

        h_y = estimate_entropy(y)
        cmi = estimate_conditional_information(x, y, exact_z)
        self.assertGreater(
            h_y / 4,
            cmi,
            'redundant copies should have little information')
