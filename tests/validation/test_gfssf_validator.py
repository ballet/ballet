import unittest

import numpy as np
import pandas as pd
import ballet.validation.gfssf_validator as gfv

class DiffCheckTest(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_discrete_entropy(self):
        same_val_arr = np.ones((50, 1))
        same_val_h = gfv._calculate_disc_entropy(same_val_arr)
        assertEquals(0, same_val_h)
        
