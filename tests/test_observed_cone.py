import unittest

import numpy as np

from scripts.cloudy.extract_ngc1068_observed_cone import (
    angular_measurement, circular_regions, output_grid, sector_mask,
)


class ObservedConeTests(unittest.TestCase):
    def test_circular_sector_crossing_zero_is_not_duplicated(self):
        support = np.zeros(120, dtype=bool)
        support[115:] = True
        support[:8] = True
        self.assertEqual(circular_regions(support), [(115, 13)])
        self.assertEqual(circular_regions(np.ones(120, dtype=bool)), [])
        self.assertEqual(circular_regions(np.zeros(120, dtype=bool)), [])

    def test_grid_and_sector_use_sky_pa_east_of_north(self):
        _, _, east, north = output_grid()
        cy, cx = np.array(east.shape)//2
        self.assertGreater(east[cy, cx-20], 0.)
        self.assertGreater(north[cy+20, cx], 0.)
        mask = sector_mask(east, north, {'pa_east_of_north_deg': 0., 'opening_half_deg': 20.}, .3, 3.)
        self.assertTrue(mask[cy+20, cx])
        self.assertFalse(mask[cy, cx-20])

    def test_known_lobes_and_uniform_image_no_fallback(self):
        y, x = np.indices((201, 201))
        east, north = (100-x)*.05, (y-100)*.05
        pa = np.rad2deg(np.arctan2(east, north)) % 360
        delta = lambda center: abs((pa-center+180)%360-180)
        signal = np.where(delta(30.) < 30., 1., -.1)
        signal = np.where(delta(210.) < 24., .5, signal)
        valid = np.ones(signal.shape, dtype=bool)
        measurement, _ = angular_measurement(signal, valid, east, north, .7, 3.7, .2)
        self.assertAlmostEqual(measurement['primary']['pa_east_of_north_deg'], 30., delta=3.)
        self.assertAlmostEqual(measurement['counter']['pa_east_of_north_deg'], 210., delta=3.)
        self.assertGreater(measurement['primary']['opening_full_deg'], 55.)
        self.assertLess(measurement['primary']['opening_full_deg'], 80.)
        flat, _ = angular_measurement(np.ones_like(signal), valid, east, north, .7, 3.7, .2)
        self.assertIsNone(flat['primary'])
        self.assertEqual(flat['status'], 'no_distinct_lobe')


if __name__ == '__main__':
    unittest.main()
