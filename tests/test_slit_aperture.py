import unittest
import numpy as np
from scipy.integrate import quad
from src.simple_utils.slit_aperture import sky_offsets,gaussian_slit_acceptance


class SlitApertureTests(unittest.TestCase):
    def test_position_angle_is_east_of_north(self):
        east=np.array([0.,1.,2.]);north=np.array([1.,0.,0.])
        values=gaussian_slit_acceptance(east,north,pa_deg=0,along_center=1,cross_center=0,width=.5,length=.5,fwhm=0)
        np.testing.assert_array_equal(values,[1,0,0])
        values=gaussian_slit_acceptance(east,north,pa_deg=90,along_center=1,cross_center=0,width=.5,length=.5,fwhm=0)
        np.testing.assert_array_equal(values,[0,1,0])

    def test_acceptance_matches_direct_psf_integration(self):
        sigma=.6/2.354820045
        def density(x,center):return np.exp(-.5*((x-center)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
        expected=quad(density,-.45,.45,args=(.2,))[0]*quad(density,-2,2,args=(-2.4,))[0]
        actual=gaussian_slit_acceptance(.2,-2.4,pa_deg=0,along_center=0,cross_center=0,width=.9,length=4,fwhm=.6)
        self.assertAlmostEqual(float(actual),expected,places=12)

    def test_uniform_surface_brightness_conserves_aperture_area(self):
        s=np.arange(-5,5,.025);e,n=np.meshgrid(s,s)
        a=gaussian_slit_acceptance(e,n,pa_deg=38,along_center=.2,cross_center=-.1,width=.9,length=3.99,fwhm=.6)
        self.assertAlmostEqual(float(a.sum()*.025**2),.9*3.99,places=8)

    def test_source_grid_center_and_scale(self):
        e,n=sky_offsets((3,3),.1,[[-1,0],[0,1]])
        self.assertAlmostEqual(e[1,2],-.1);self.assertAlmostEqual(n[2,1],.1)
        self.assertEqual(e[1,1],0);self.assertEqual(n[1,1],0)


if __name__=='__main__':unittest.main()
