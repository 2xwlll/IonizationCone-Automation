"""Physical checks for passband integration, not renderer implementation copies."""

from pathlib import Path
import unittest

import numpy as np

from src.simple_utils.synthetic_photometry import (
    HC_ERG_ANGSTROM, bandpass_coverage, integrate_bandpass, photon_bandpass_weights,
)


class PhotonPhotometryTests(unittest.TestCase):
    def test_flat_continuum_has_analytic_response_without_endpoint_overcount(self):
        wave = np.array([4900.0, 4900.1, 4957.0, 5050.0, 5100.0])
        fw = np.array([4900.0, 5100.0])
        response = np.full(2, 0.25)
        expected = 0.25*(5100.0**2-4900.0**2)/(2*HC_ERG_ANGSTROM)
        weights = photon_bandpass_weights(wave, fw, response)
        self.assertAlmostEqual(weights.sum()/expected, 1.0, places=13)

    def test_photon_conversion_matches_monochromatic_energy(self):
        # A Gaussian line of unit integrated energy should give lambda/(h*c)
        # photons at constant throughput, up to vanishing Gaussian tail loss.
        wave = np.linspace(4900, 5100, 4001)
        center, sigma = 5007.0, 1.4
        line = np.exp(-0.5*((wave-center)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        weights = photon_bandpass_weights(wave, np.array([4900., 5100.]),
                                          np.array([0.3, 0.3]))
        self.assertAlmostEqual(float(weights@line)/(0.3*center/HC_ERG_ANGSTROM),
                               1.0, places=12)

    def test_real_filter_lines_and_velocity_shifts_match_fine_reference(self):
        curve = np.loadtxt(Path("configs/instruments/hst_wfpc2_pc_f502n.csv"), delimiter=",")
        fw, response = curve.T
        wave = np.arange(4960, 5065.01, 0.5)
        weights = photon_bandpass_weights(wave, fw, response)
        fine = np.linspace(4960, 5065, 105001)
        fine_response = np.interp(fine, fw, response, left=0, right=0)
        z, c = 0.003793, 299792.458
        measured = {}
        for rest in (4958.91, 5006.84):
            for velocity in (-900., 0., 900.):
                center = rest*(1+z)*(1+velocity/c)
                sigma = rest*(1+z)*85/c
                coarse_line = np.exp(-0.5*((wave-center)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
                fine_line = np.exp(-0.5*((fine-center)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
                expected = np.trapezoid(fine_line*fine_response*fine/HC_ERG_ANGSTROM, fine)
                actual = float(weights@coarse_line)
                # Check relative sampling error when the filter passes >=1e-5
                # of this line. At virtually zero transmission a relative
                # error is ill-conditioned; bound absolute efficiency instead.
                efficiency = expected*HC_ERG_ANGSTROM/center
                if efficiency >= 1e-5:
                    self.assertLess(abs(actual/expected-1), 0.006)
                else:
                    self.assertLess(abs(actual-expected)*HC_ERG_ANGSTROM/center, 1e-7)
                measured[(rest, velocity)] = actual
        self.assertGreater(measured[(5006.84, 0)]/measured[(4958.91, 0)], 200)
        self.assertLess(measured[(5006.84, 900)], measured[(5006.84, 0)])

    def test_legacy_cube_coverage_is_rejected(self):
        fw, response = np.loadtxt("configs/instruments/hst_wfpc2_pc_f502n.csv", delimiter=",").T
        wave = np.linspace(4965, 5040, 96)
        self.assertLess(bandpass_coverage(wave, fw, response), 0.999)
        with self.assertRaisesRegex(ValueError, "extend the internal spectrum"):
            integrate_bandpass(np.ones((96, 1, 1)), wave, fw, response)
        extended = np.linspace(4960, 5065, 211)
        self.assertAlmostEqual(bandpass_coverage(extended, fw, response), 1, places=13)

    def test_cube_keeps_spatial_morphology_and_linearity(self):
        wave = np.array([4990., 5000., 5020., 5040.])
        fw = np.array([4980., 5000., 5020., 5050.])
        response = np.array([0., 0.05, 0.05, 0.])
        spectrum = np.array([0.2, 1., 2., 0.3])
        morphology = np.array([[1., 3.], [0., 2.]])
        cube = spectrum[:, None, None]*morphology
        image, weights, _ = integrate_bandpass(cube, wave, fw, response, minimum_coverage=0)
        np.testing.assert_allclose(image, morphology*(weights@spectrum), rtol=1e-13)
        doubled, _, _ = integrate_bandpass(2*cube, wave, fw, response, minimum_coverage=0)
        np.testing.assert_allclose(doubled, 2*image, rtol=1e-13)
        self.assertEqual(image.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
