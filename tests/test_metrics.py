import unittest

import numpy as np

import searchnets


class TestMetrics(unittest.TestCase):
    def test_p_item_grid(self):
        g1 = [['t', ''], ['', '']]
        g2 = [['t', ''], ['', '']]
        g3 = [['t', ''], ['', '']]
        g4 = [['', ''], ['', 't']]
        char_grids = [g1, g2, g3, g4]

        p = searchnets.utils.metrics.p_item_grid(char_grids)
        expected_p = np.asarray([[0.75, 0.], [0., 0.25]])
        self.assertTrue(
            np.array_equal(p, expected_p)
        )

        g5 = [['', '', ''], ['', 't', '']]
        char_grids_multiple_shapes = [g1, g2, g5]
        with self.assertRaises(ValueError):
            searchnets.utils.metrics.p_item_grid(char_grids_multiple_shapes)


if __name__ == '__main__':
    unittest.main()
