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

    def test_compute_d_prime(self):
        y_true = np.random.choice(np.asarray([0, 1]), size=(500,))
        y_pred = np.copy(y_true)
        hit_rate, false_alarm_rate, d_prime = searchnets.utils.metrics.compute_d_prime(y_true, y_pred)
        # range of d_prime is, roughly, (-5.7538..., 5.75388...)
        self.assertTrue(d_prime > 5.75)

        y_pred_not = np.logical_not(y_true).astype(int)
        hr_not, far_not, d_prime_not = searchnets.utils.metrics.compute_d_prime(y_true, y_pred_not)
        self.assertTrue(d_prime_not < -5.75)


if __name__ == '__main__':
    unittest.main()

