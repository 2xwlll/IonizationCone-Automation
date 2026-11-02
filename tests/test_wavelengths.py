import unittest
import numpy as np
from src.simple_utils.wavelengths import cloudy_air_to_vacuum


class WavelengthConventionTests(unittest.TestCase):
    def test_matches_cloudy_c25_peck_reeder_reference_values(self):
        np.testing.assert_allclose(cloudy_air_to_vacuum([5006.84,18751.]),
                                   [5008.236595830281,18756.119850349412],rtol=1e-13)

    def test_short_cloudy_lines_already_use_vacuum(self):
        np.testing.assert_array_equal(cloudy_air_to_vacuum([1215.67,1640.]),[1215.67,1640.])

    def test_invalid_wavelength_is_rejected(self):
        with self.assertRaises(ValueError):cloudy_air_to_vacuum(0.)


if __name__=='__main__':unittest.main()
