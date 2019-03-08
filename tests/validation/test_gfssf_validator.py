import unittest

import numpy as np
import pandas as pd

import ballet.validation.gfssf_validator as gfv


class DiffCheckTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_discrete_entropy(self):
        same_val_arr_ones = np.ones((50, 1))
        same_val_h = gfv._calculate_disc_entropy(same_val_arr_ones)
        self.assertEqual(0, same_val_h, 'Expected all ones to be zero entropy')

        same_val_arr_zero = np.zeros((50, 1))
        same_multi_val_arr = np.concatenate(
            (same_val_arr_ones, same_val_arr_zero), axis=1)
        same_multi_val_h = gfv._calculate_disc_entropy(same_multi_val_arr)
        self.assertEqual(
            0,
            same_multi_val_h,
            'Expected all ones to be zero entropy')

        diff_val_arr = np.concatenate((same_val_arr_ones, same_val_arr_zero))
        expected_h = np.log(0.5)
        diff_val_h = gfv._calculate_disc_entropy(diff_val_arr)
        self.assertAlmostEqual(
            expected_h,
            diff_val_h,
            'Expected all some entropy in Ber(0.5)')
