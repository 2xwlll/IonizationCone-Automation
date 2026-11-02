"""Analytic fixtures for the published-line fitting machinery, not galaxy data."""
import unittest
import numpy as np
from scripts.cloudy.fit_ngc1068_spectral_constraints import FIT_IDS,REF,fit_lines
from src.simple_utils.wavelengths import cloudy_air_to_vacuum


class SpectralConstraintTests(unittest.TestCase):
    def test_recovers_known_scale_and_attenuation_without_flux_unit_underflow(self):
        waves=np.array([12818.1,21655.3,10938.1,10123.3,12566.8,16435.5,19624.7])
        intrinsic=np.array([.002,.0003,.001,.0004,.0007,.0005,.0008])
        grid={'line_names':np.array(FIT_IDS),'rest_line_angstrom':waves,
              'line_flux':intrinsic[None,:],'model_id':np.array(['analytic_fixture']),
              'log_u':np.array([-2.25]),'log_nh_cm3':np.array([3.])}
        tau=.5;scale=1200.
        flux=scale*intrinsic*np.exp(-tau*(cloudy_air_to_vacuum(waves)/REF)**(-1.3))
        measurements={name:{'flux_1e13':float(value),'flux_error_1e13':float(.05*value)}
                      for name,value in zip(FIT_IDS,flux)}
        result=fit_lines(grid,measurements)[0]
        self.assertAlmostEqual(result['tau_at_pa_alpha'],tau,places=12)
        self.assertAlmostEqual(result['scale_in_table_flux_units_per_cloudy_line_unit'],scale,places=8)
        self.assertLess(result['weighted_residual_sum'],1e-20)


if __name__=='__main__':unittest.main()
