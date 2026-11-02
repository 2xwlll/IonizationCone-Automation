import unittest
import numpy as np
from src.simple_utils.nuclear_sed import nuclear_flambda_shape,nuclear_filter_response
from src.simple_utils.synthetic_photometry import photon_bandpass_weights


class NuclearSEDTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'reference_rest_angstrom':18751.,'hot_dust_temperature_k':1100.,
                  'hot_dust_fraction_at_reference':.9,'powerlaw_alpha_nu':-.5,
                  'tau_at_reference':1.,'extinction_exponent':1.3}

    def test_normalization_and_finite_near_ir_spectrum(self):
        self.assertAlmostEqual(float(nuclear_flambda_shape(18751.,self.cfg)),1.)
        values=nuclear_flambda_shape(np.linspace(6000,55000,1000),self.cfg)
        self.assertTrue(np.isfinite(values).all() and (values>0).all())

    def test_flat_flambda_source_has_correct_redshift_jacobian(self):
        cfg={**self.cfg,'hot_dust_fraction_at_reference':0.,'powerlaw_alpha_nu':-2.,'tau_at_reference':0.}
        wave=np.linspace(18000,19500,101);t=np.ones_like(wave)*.4;z=.1
        expected=photon_bandpass_weights(wave,wave,t).sum()/(1+z)
        self.assertAlmostEqual(nuclear_filter_response(wave,t,z,cfg)/expected,1.,places=12)

    def test_hotter_dust_has_bluer_normalized_color(self):
        a=nuclear_flambda_shape(12000.,self.cfg)
        b=nuclear_flambda_shape(12000.,{**self.cfg,'hot_dust_temperature_k':1500.})
        self.assertGreater(b,a)


if __name__=='__main__':unittest.main()
