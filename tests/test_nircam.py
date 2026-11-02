"""Physical checks of the NIRCam response foundation, using cached official data."""

import unittest

import numpy as np

from src.jwst_utils.nircam import load_filters, redshift_flambda, render_filters
from src.simple_utils.synthetic_photometry import integrate_bandpass


class NIRCamTests(unittest.TestCase):
    def test_redshift_preserves_integrated_energy_and_moves_line(self):
        wave = np.linspace(4900, 5100, 1001)
        line = np.exp(-0.5*((wave-5007)/2)**2)[:, None, None]
        shifted, observed = redshift_flambda(line, wave, 2.0)
        self.assertAlmostEqual(np.trapezoid(shifted[:, 0, 0], observed),
                               np.trapezoid(line[:, 0, 0], wave), places=12)
        self.assertEqual(observed[np.argmax(shifted)], 15021)

    def test_all_29_filters_preserve_photometry_and_common_morphology(self):
        # A delta PSF makes this a check of absolute weighting and linearity.
        wave = np.linspace(2500, 60000, 301)
        morphology = np.array([[0., 1.], [2., 4.]])
        cube = np.ones((len(wave), 1, 1))*morphology
        identity = lambda name, wavelength, pixel_scale: np.ones((1, 1))
        images, metadata = render_filters(cube, wave, redshift=0.5,
                                         pixel_scale_arcsec=0.005, kernel_provider=identity)
        self.assertEqual(len(images), 29)
        shifted, observed = redshift_flambda(cube, wave, 0.5)
        for name, entry in load_filters().items():
            fw, throughput = entry["curve"].T
            expected, _, _ = integrate_bandpass(shifted, observed, fw, throughput)
            np.testing.assert_allclose(images[name], expected, atol=1e-6, rtol=1e-13)
            self.assertGreaterEqual(metadata[name]["coverage"], 0.999)

    def test_chromatic_blur_spreads_point_light_and_preserves_response(self):
        widths = []
        for name in ("F070W", "F444W"):
            entry = load_filters()[name]
            wave, throughput = entry["curve"].T
            cube = np.zeros((len(wave), 161, 161), dtype=np.float32)
            cube[:, 80, 80] = 1
            images, _ = render_filters(cube, wave, redshift=0,
                                       pixel_scale_arcsec=0.005, filters=[name])
            image = images[name]
            expected, _, _ = integrate_bandpass(cube[:, 80:81, 80:81], wave, wave, throughput)
            self.assertAlmostEqual(image.sum()/expected.item(), 1, places=6)
            x = np.arange(161)-80
            widths.append(np.sqrt(np.sum(image*x[None, :]**2)/image.sum()))
            self.assertGreater(np.count_nonzero(image > image.max()*0.01), 1)
        self.assertGreater(widths[1]/widths[0], 5)

    def test_old_optical_cube_cannot_be_used_as_a_full_nircam_spectrum(self):
        wave = np.linspace(4960, 5065, 211)
        with self.assertRaisesRegex(ValueError, "extend the rest-frame spectrum"):
            render_filters(np.ones((211, 1, 1)), wave, redshift=0.003793,
                           pixel_scale_arcsec=0.005)

    def test_negative_redshift_and_unknown_filter_rejected(self):
        with self.assertRaises(ValueError):
            redshift_flambda(np.ones((2, 1, 1)), [5000, 6000], -1)
        with self.assertRaisesRegex(ValueError, "Unknown NIRCam filter"):
            render_filters(np.ones((2, 1, 1)), [2000, 60000], redshift=0,
                           pixel_scale_arcsec=0.005, filters=["F502N"])


if __name__ == "__main__":
    unittest.main()
