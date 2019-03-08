import unittest

import numpy as np
import pandas as pd

import ballet.validation.gfssf_validator as gfv


class GFSSFValidadatorTest(unittest.TestCase):
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
            msg='Expected all ones to be zero entropy')

        diff_val_arr = np.concatenate((same_val_arr_ones, same_val_arr_zero))
        expected_h = np.log(2)
        diff_val_h = gfv._calculate_disc_entropy(diff_val_arr)
        self.assertAlmostEqual(
            expected_h,
            diff_val_h,
            msg='Expected entropy in x ~ Ber(0.5)')
    
    def test_conditional_mutual_information(self):
        # redundant copies have little information
        x = np.reshape(np.arange(1,101), (100,1))
        y = np.reshape(np.arange(1,101), (100,1))
        h_y = gfv._estimate_entropy(y)

        # exact copies of y should have lots of information
        useless_z = np.ones((100,1))
        mi = gfv._estimate_conditional_information(x,y,useless_z)
        self.assertGreater(mi, h_y / 4, 'exact, non-redundant copies should have little information')

        exact_z = np.reshape(np.arange(1,101), (100,1))
        mi = gfv._estimate_conditional_information(x,y,exact_z)
        self.assertGreater(h_y / 4, mi, 'redundant copies should have little information')


