import unittest

import numpy as np

from ballet.util import asarray2d
from ballet.validation.entropy import (
    estimate_disc_entropy, estimate_conditional_information, estimate_entropy,
    estimate_mutual_information)


class EntropyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_discrete_entropy_all_ones(self):
        same_val_arr_ones = np.ones((50, 1))
        same_val_h = estimate_disc_entropy(same_val_arr_ones)
        self.assertEqual(0, same_val_h, 'Expected all ones to be zero entropy')

    def test_discrete_entropy_all_same(self):
        same_val_arr_zero = np.zeros((50, 1))
        same_val_arr_ones = np.ones((50, 1))
        same_multi_val_arr = np.concatenate(
            (same_val_arr_ones, same_val_arr_zero), axis=1)
        same_multi_val_h = estimate_disc_entropy(same_multi_val_arr)
        self.assertEqual(
            0,
            same_multi_val_h,
            msg='Expected all same values to be zero entropy')

    def test_discrete_entropy_two_values(self):
        same_val_arr_zero = np.zeros((50, 1))
        same_val_arr_ones = np.ones((50, 1))
        # Test on a dataset that mimics a fair coin (half ones, half zeros)
        diff_val_arr = np.concatenate(
            (same_val_arr_ones, same_val_arr_zero), axis=0)
        expected_h = np.log(2)
        diff_val_h = estimate_disc_entropy(diff_val_arr)
        self.assertAlmostEqual(
            expected_h,
            diff_val_h,
            msg='Expected entropy in x ~ Ber(0.5)')

    def test_entropy_cont_disc_heuristics(self):
        arange_disc_arr = asarray2d(np.arange(50))
        arange_cont_arr = asarray2d(np.arange(50) + 0.5)

        disc_h = estimate_entropy(arange_disc_arr)
        cont_h = estimate_entropy(arange_cont_arr)
        self.assertNotEqual(
            disc_h,
            cont_h,
            msg='Expected cont, disc columns to be handled differently')

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
        x = np.reshape(np.arange(1, 101), (100, 1))
        y = np.ones((100, 1))
        mi = estimate_mutual_information(x, y)
        h_z = estimate_entropy(x)
        self.assertGreater(
            h_z / 4,
            mi,
            'uninformative column should have no information')

    def test_mi_informative(self):
        x = np.reshape(np.arange(1, 101), (100, 1))
        y = np.reshape(np.arange(1, 101), (100, 1))
        mi = estimate_mutual_information(x, y)
        h_y = estimate_entropy(y)
        self.assertGreater(
            mi,
            h_y / 4,
            'exact copy columns should have high information')

    def test_cmi_high_info_uninformative_z(self):
        # redundant copies have little information
        x = np.reshape(np.arange(1, 101), (100, 1))
        y = np.reshape(np.arange(1, 101), (100, 1))

        # exact copies of y should have lots of information
        useless_z = np.ones((100, 1))
        cmi = estimate_conditional_information(x, y, useless_z)
        mi = estimate_mutual_information(x, y)
        self.assertAlmostEqual(
            cmi,
            mi,
            'uninformative z should not affect mutual information score')

    def test_cmi_redundant_info(self):
        x = np.reshape(np.arange(1, 101), (100, 1))
        y = np.reshape(np.arange(1, 101), (100, 1))
        exact_z = np.reshape(np.arange(1, 101), (100, 1))

        h_y = estimate_entropy(y)
        cmi = estimate_conditional_information(x, y, exact_z)
        self.assertGreater(
            h_y / 4,
            cmi,
            'redundant copies should have little information')
