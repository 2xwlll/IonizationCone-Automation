"""Physical and saved-output checks for custom filters on existing narrow cubes."""

import json
from math import erf, sqrt, pi
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.simple_utils.synthetic_filters import filter_cube, gaussian_passband, top_hat_passband
from src.simple_utils.synthetic_photometry import HC_ERG_ANGSTROM


class SyntheticFilterTests(unittest.TestCase):
    def test_rectangle_exact_edges_between_nonuniform_source_samples(self):
        wave = np.array([4900., 4981., 5003., 5050., 5100.])
        morphology = np.array([[1., 0.], [0., 3.]])
        cube = np.ones((5, 1, 1))*morphology
        fw, response = top_hat_passband(4992.25, 5037.75, 0.4)
        image, _, coverage = filter_cube(cube, wave, fw, response)
        expected = 0.4*(5037.75**2-4992.25**2)/(2*HC_ERG_ANGSTROM)
        np.testing.assert_allclose(image, expected*morphology, rtol=1e-13)
        self.assertAlmostEqual(coverage, 1, places=12)

    def test_gaussian_matches_analytic_flat_spectrum_response(self):
        center, fwhm = 5025., 10.
        wave = np.linspace(4980, 5070, 181)
        fw, response = gaussian_passband(center, fwhm)
        image, _, _ = filter_cube(np.ones((181, 1, 1)), wave, fw, response)
        sigma = fwhm/np.sqrt(8*np.log(2))
        expected = center*sigma*sqrt(2*pi)*erf(4/sqrt(2))/HC_ERG_ANGSTROM
        self.assertLess(abs(image.item()/expected-1), 1e-6)

    def test_line_selection_keeps_spatial_positions_and_linear_brightness(self):
        wave = np.linspace(4900, 5100, 801)
        cube = np.zeros((801, 5, 5))
        cube[:, 1, 1] = np.exp(-0.5*((wave-4960)/1.5)**2)
        cube[:, 3, 3] = 2*np.exp(-0.5*((wave-5030)/1.5)**2)
        fw, response = top_hat_passband(5020, 5040)
        image, _, _ = filter_cube(cube, wave, fw, response)
        self.assertEqual(np.argmax(image), np.ravel_multi_index((3, 3), image.shape))
        self.assertLess(image[1, 1], image[3, 3]*1e-100)
        background = np.ones(image.shape, dtype=bool)
        background[1, 1] = background[3, 3] = False
        np.testing.assert_array_equal(image[background], 0)
        doubled, _, _ = filter_cube(2*cube, wave, fw, response)
        np.testing.assert_allclose(doubled, 2*image, rtol=1e-13)

    def test_passband_outside_cube_is_rejected_even_for_small_missing_tail(self):
        wave = np.linspace(5000, 5050, 101)
        fw, response = gaussian_passband(5025, 15)
        with self.assertRaisesRegex(ValueError, "outside the cube"):
            filter_cube(np.ones((101, 1, 1)), wave, fw, response)

    def test_cli_retains_existing_psf_redshift_mask_and_linear_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root/'cube.npz', root/'image.npz'
            wave = np.linspace(4960, 5065, 211)
            cube = np.ones((211, 3, 3))
            cube[:, 1, 1] *= 7
            mask = np.eye(3, dtype=np.float32)
            np.savez(source, input=cube, mask=mask, wavelength_angstrom=wave)
            metadata = {"wavelength_frame": "observed", "redshift": 0.003793,
                        "psf_fwhm_arcsec": 0.075,
                        "normalization": {"transform": "linear"}}
            source.with_suffix('.json').write_text(json.dumps(metadata))
            command = [sys.executable, 'scripts/cloudy/render_synthetic_filter.py',
                       str(source), '--output', str(output), '--tophat', '5005', '5047']
            subprocess.run(command, check=True, capture_output=True, text=True)
            with np.load(output, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['mask'], mask)
                np.testing.assert_allclose(saved['image'], np.tensordot(saved['weights'], cube, axes=(0, 0)))
                self.assertAlmostEqual(saved['image'][1, 1]/saved['image'][0, 0], 7)
                record = json.loads(saved['metadata'].item())
                self.assertEqual(record['source_metadata'], metadata)
            overwrite = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(overwrite.returncode, 0)
            self.assertIn('already exists', overwrite.stderr)


if __name__ == '__main__':
    unittest.main()
