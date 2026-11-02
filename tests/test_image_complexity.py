import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from src.simple_utils.image_complexity import component_metrics, psf_axis_fwhm
from scripts.cloudy.measure_f187n_complexity import load_components


class ImageComplexityTests(unittest.TestCase):
    def measure(self, c, b, **kwargs):
        return component_metrics(c, b, np.ones(c.shape, dtype=bool),
                                 pixel_scale_arcsec=.1, **kwargs)

    def test_known_mixture_and_scale_invariance(self):
        c = np.ones((31, 31)) * 2
        b = np.ones_like(c) * 3
        for scale in [1., 17.]:
            m = self.measure(c * scale, b * scale, noise_sigma=np.ones_like(c) * scale)
            self.assertAlmostEqual(m['other_over_cone'], 1.5)
            self.assertAlmostEqual(m['cone_fraction'], .4)
            self.assertEqual(m['other_dominant_area_fraction'], 1.)
            self.assertAlmostEqual(m['structure'][1]['other'], 0.)
            self.assertAlmostEqual(m['median_cone_signal_over_pixel_sigma'], 2.)

    def test_same_flux_different_structure(self):
        c = np.ones((65, 65))
        smooth = np.ones_like(c)
        knots = np.zeros_like(c)
        knots[::4, ::4] = 1.
        knots *= smooth.sum() / knots.sum()
        a, b = self.measure(c, smooth), self.measure(c, knots)
        self.assertAlmostEqual(a['other_over_cone'], b['other_over_cone'])
        self.assertGreater(b['structure'][1]['other'], a['structure'][1]['other'] + .5)

    def test_other_amplitude_changes_contamination_but_not_normalized_structure(self):
        y, x = np.indices((65, 65))
        c = np.ones_like(x, dtype=float)
        b = 1. + np.exp(-((x - 32)**2 + (y - 32)**2) / 8.)
        first, second = self.measure(c, b), self.measure(c, 4. * b)
        self.assertAlmostEqual(second['other_over_cone'], 4. * first['other_over_cone'])
        self.assertAlmostEqual(first['structure'][1]['other'], second['structure'][1]['other'])

    def test_weighted_areas(self):
        c = np.ones((2, 2))
        b = np.array([[2., 0.], [0., 0.]])
        m = self.measure(c, b, pixel_area=np.array([[3., 1.], [1., 1.]]))
        self.assertAlmostEqual(m['other_over_cone'], 1.)
        self.assertAlmostEqual(m['other_dominant_area_fraction'], .5)

    def test_no_cone_and_empty_aperture_are_explicit(self):
        c = np.zeros((9, 9))
        m = self.measure(c, np.ones_like(c))
        self.assertEqual(m['status'], 'zero_cone_signal')
        self.assertIsNone(m['other_over_cone'])
        self.assertEqual(m['cone_fraction'], 0.)
        m = component_metrics(c, c, c.astype(bool), pixel_scale_arcsec=.1)
        self.assertEqual(m['status'], 'empty_aperture')
        self.assertIsNone(m['cone_fraction'])
        self.assertIsNone(m['other_dominant_area_fraction'])

    def test_aperture_is_not_a_smoothing_boundary(self):
        c = np.ones((51, 51))
        aperture = np.zeros_like(c, dtype=bool)
        aperture[10:40, 10:40] = True
        m = component_metrics(c, c, aperture, pixel_scale_arcsec=.1)
        self.assertAlmostEqual(m['structure'][2]['cone'], 0.)

    def test_invalid_inputs_and_partial_noise_coverage(self):
        c = np.ones((9, 9))
        for bad in [-c, c * np.nan]:
            with self.assertRaises(ValueError):
                self.measure(c, bad)
        with self.assertRaises(ValueError):
            self.measure(c, c, smoothing_sigmas_arcsec=[0.])
        sigma = np.ones_like(c)
        sigma[0] = np.nan
        m = self.measure(c, c, noise_sigma=sigma)
        self.assertEqual(m['noise_valid_pixels'], 72)
        self.assertEqual(m['median_cone_signal_over_pixel_sigma'], 1.)

    def test_psf_width_matches_analytic_gaussian(self):
        y, x = np.indices((201, 201))
        kernel = np.exp(-.5 * (((x - 100) / 12.)**2 + ((y - 100) / 8.)**2))
        widths = psf_axis_fwhm(kernel, .01)
        self.assertAlmostEqual(widths['x_arcsec'], 2.35482 * 12 * .01, delta=.0002)
        self.assertAlmostEqual(widths['y_arcsec'], 2.35482 * 8 * .01, delta=.0002)
        with self.assertRaises(ValueError):
            psf_axis_fwhm(np.ones((9, 9)), .01)


class SavedComponentTests(unittest.TestCase):
    def archive(self, directory, *, inconsistent=False, duplicate=False, negative=False):
        path = Path(directory) / 'image_mask.npz'
        c = np.ones((9, 9)) * 2.
        b = np.ones_like(c) * 3.
        if negative:
            b[0, 0] = -1.
        names = ['cone_pa_alpha', 'stellar_continuum']
        if duplicate:
            names.append('cone_pa_alpha')
        metadata = {'filter': 'F187N', 'component_names': names,
                    'aggregate_names': ['pa_alpha'],
                    'galaxy_plus_background_baseline_mjysr': 10.}
        np.savez(path, cone_pa_alpha=c, stellar_continuum=b,
                 pa_alpha=c, clean=c+b+(11. if inconsistent else 10.),
                 input=c+b+10., mask=np.ones_like(c), reference_valid=np.ones_like(c, dtype=bool),
                 reference_error=np.ones_like(c), metadata=json.dumps(metadata))
        return path

    def test_saved_aggregate_is_not_double_counted_and_baseline_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            _, c, b, _, _, _, _, checks = load_components(self.archive(directory))
        self.assertTrue(np.all(c == 2.))
        self.assertTrue(np.all(b == 3.))
        self.assertEqual(checks['baseline_excluded'], 10.)
        self.assertEqual(checks['reconstruction_max_absolute_error'], 0.)

    def test_inconsistent_duplicate_and_negative_components_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for issue in ['inconsistent', 'duplicate', 'negative']:
                with self.subTest(issue=issue), self.assertRaises(ValueError):
                    load_components(self.archive(directory, **{issue: True}))


if __name__ == '__main__':
    unittest.main()
