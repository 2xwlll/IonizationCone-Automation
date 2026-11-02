"""Physical checks for the F187N pilot's spectral and spatial operations."""

import unittest

import numpy as np

from src.simple_utils.f187n import (
    line_response, printed_emergent_lines, rebin_surface_brightness, resample_psf,
)
from src.simple_utils.synthetic_photometry import HC_ERG_ANGSTROM


class F187NPilotTests(unittest.TestCase):
    def test_line_photon_response_matches_observed_photon_energy(self):
        rest, z = 18751., .003793
        center = rest*(1+z)
        result = line_response(rest,z,0,85,np.array([17000.,21000.]),np.array([.4,.4]))
        expected = .4*center/HC_ERG_ANGSTROM
        self.assertAlmostEqual(result/expected,1,places=12)

    def test_final_emergent_table_replaces_intrinsic_values_and_keeps_missing_unknown(self):
        text = ('Intrinsic line intensities\nH  1  1.87510m  -2.000  1.0000\n'
                'Emergent line intensities\nH  1  1.87510m  -3.000  0.3000'
                '      He 1  1.86854m  -5.000  0.0030\n')
        found = printed_emergent_lines(text)
        self.assertEqual(found[('H 1',18751.)],.001)
        self.assertEqual(found[('He 1',18685.4)],.00001)
        self.assertNotIn(('He 2',18757.),found)

    def test_psf_resampling_preserves_center_and_angular_width(self):
        yy,xx = np.indices((81,81))
        image = np.exp(-((xx-40)**2+(yy-40)**2)/(2*5**2))
        image /= image.sum()
        sampled = resample_psf(image,.01,.007)
        self.assertAlmostEqual(sampled.sum(),1,places=12)
        sy,sx = np.indices(sampled.shape)
        center = (sampled.shape[0]-1)/2
        self.assertAlmostEqual(float((sampled*sx).sum()),center,places=10)
        sigma = np.sqrt((sampled*(sx-center)**2).sum())*.007
        self.assertLess(abs(sigma/.05-1),.01)

    def test_detector_rebin_preserves_surface_brightness_and_integrated_sky_flux(self):
        image = np.arange(144,dtype=float).reshape(12,12)
        rebinned = rebin_surface_brightness(image,3)
        self.assertEqual(rebinned.shape,(4,4))
        self.assertAlmostEqual(image.sum()*.01**2,rebinned.sum()*.03**2,places=12)
        np.testing.assert_allclose(rebin_surface_brightness(np.ones((12,12))*7,3),7)


if __name__=='__main__':
    unittest.main()
