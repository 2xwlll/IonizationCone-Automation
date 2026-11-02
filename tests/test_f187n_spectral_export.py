"""Independent numerical checks of spectral extraction, not galaxy validation."""
import unittest
import numpy as np
from scripts.cloudy.plot_f187n_spectral_reference import velocity_histogram, integrated_profile
from src.jwst_utils.nircam import convolve_kernel
from src.simple_utils.f187n import line_response
from src.simple_utils.synthetic_photometry import photon_bandpass_weights


class SpectralExportTests(unittest.TestCase):
    def test_velocity_histogram_is_adjoint_of_response_interpolation(self):
        bins=np.linspace(-1200,1200,241)
        velocity=np.array([[-1300.,-115.,73.],[23.,812.,1300.]])
        energy=np.array([[1.,2.,4.],[8.,16.,32.]])
        response=np.sin(bins/200.)+2
        expected=np.sum(energy*np.interp(velocity,bins,response))
        actual=velocity_histogram(velocity,energy,bins)@response
        self.assertAlmostEqual(actual,expected,places=12)

    def test_adjoint_psf_preserves_finite_field_sum_with_asymmetric_kernel(self):
        image=np.arange(143,dtype=float).reshape(11,13)
        kernel=np.array([[0.,1.,2.],[3.,5.,1.],[0.,1.,0.]])
        acceptance=convolve_kernel(np.ones_like(image),kernel[::-1,::-1])
        self.assertAlmostEqual(float(np.sum(image*acceptance)),float(convolve_kernel(image,kernel).sum()),places=9)

    def test_detector_aperture_adjoint_includes_rebin_and_invalid_pixel_holes(self):
        from src.simple_utils.f187n import rebin_surface_brightness
        image=np.arange(144,dtype=float).reshape(12,12)
        aperture=np.zeros((6,6));aperture[1:5,1:5]=1;aperture[2,3]=0
        kernel=np.array([[0.,1.,2.],[3.,5.,1.],[0.,1.,0.]])
        high=np.repeat(np.repeat(aperture,2,axis=0),2,axis=1)
        acceptance=convolve_kernel(high,kernel[::-1,::-1])/4
        expected=np.sum(aperture*rebin_surface_brightness(convolve_kernel(image,kernel),2))
        self.assertAlmostEqual(float(np.sum(image*acceptance)),float(expected),places=10)

    def test_integrated_profile_retains_redshifted_line_photon_response(self):
        rest=18756.11985; z=.003793
        wave=np.linspace(18500,19100,12001)
        fw=np.linspace(18500,19100,101); t=np.linspace(.1,.6,101)
        velocities=np.array([-300.,0.,300.]); energies=np.array([1.,4.,2.])
        profile=integrated_profile(wave,rest,z,85.,velocities,energies)
        expected=sum(e*line_response(rest,z,v,85.,fw,t) for e,v in zip(energies,velocities))
        actual=float(photon_bandpass_weights(wave,fw,t)@profile)
        self.assertAlmostEqual(actual/expected,1.,places=7)


if __name__=='__main__':unittest.main()
