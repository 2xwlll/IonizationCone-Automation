import tempfile
from pathlib import Path
import unittest
import numpy as np
from scripts.cloudy.build_hii_f187n_grid import final_saved_lines,LINES,model_input
from src.simple_utils.hii_population import render_hii_population


class HIIModelTests(unittest.TestCase):
    def test_inputs_use_stellar_source_and_request_iterations(self):
        model={'model_id':'check','stellar_temperature_k':35000.,'log_u':-3.,
               'log_nh_cm3':2.,'metallicity_solar':1.,'dust_scale':1.}
        text=model_input(model,{'log_column_density_cm2':21.5})
        self.assertIn('blackbody 35000.0 K',text)
        self.assertNotIn('table agn',text)
        self.assertIn('iterate to convergence',text)

    def test_first_iteration_saved_flux_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lines.lin'
            values='\t'.join(['1.0']*len(LINES))
            path.write_text('iteration 1\t'+values+'\n')
            with self.assertRaises(ValueError):final_saved_lines(path)
            path.write_text('iteration 3\t'+values+'\n')
            iteration,flux=final_saved_lines(path)
            self.assertEqual(iteration,3)
            np.testing.assert_array_equal(flux,np.ones(len(LINES)))

    def test_complete_flat_passband_line_response_scales_with_photon_wavelength(self):
        # Analytic flat-throughput test fixture, not CLOUDY calculation output.
        wave=np.linspace(17000,21000,101)
        grid={'line_flux':np.array([[1.]]),'line_labels':np.array(['H 1']),
              'rest_line_angstrom':np.array([18751.]),'filter_wavelength_angstrom':wave,
              'throughput':np.ones_like(wave)*.4,'continuum_rest_flambda':np.zeros((1,len(wave)))}
        y,x=np.indices((31,31));x=(x-15)/10;y=(y-15)/10
        cfg={'model_index':0,'position_angle_east_of_north_deg':90.,'axis_ratio':.6,
             'profile':'disk','scale_radius_arcsec':1.,'rotation_kms':0.,'sigma_kms':30.}
        def render(z):
            return render_hii_population(grid,cfg,x,y,z,np.zeros_like(x),1.3,1,np.random.default_rng(9))['pa_alpha']
        a,b=render(0),render(.01)
        self.assertTrue((a>0).all())
        np.testing.assert_allclose(b/a,1.01,rtol=1e-12)


if __name__=='__main__':unittest.main()
