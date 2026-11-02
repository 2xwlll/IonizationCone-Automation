"""Check empirical summary fitting independently of the galaxy renderer."""
import unittest
import numpy as np
from src.simple_utils.f187n_calibration import annular_means, fit_annular_templates


class AnnularCalibrationTests(unittest.TestCase):
    def test_recovers_known_nonnegative_mixture_without_using_masked_pixels(self):
        y,x=np.indices((101,101))
        r=np.hypot(x-50,y-50)/10
        templates=np.array([np.exp(-r),np.exp(-r*r/4)])
        observed=3*templates[0]+7*templates[1]+2
        valid=np.ones(r.shape,dtype=bool)
        valid[:50,:50]=False
        observed[~valid]=1e12
        coef,_,_,_=fit_annular_templates(templates,observed,valid,r,np.linspace(.2,4.5,18))
        np.testing.assert_allclose(coef,[3,7,2],rtol=1e-10,atol=1e-10)
        fixed,_,_,_=fit_annular_templates(templates,observed,valid,r,np.linspace(.2,4.5,18),fixed_first=3.)
        np.testing.assert_allclose(fixed,[3,7,2],rtol=1e-10,atol=1e-10)
        bounded,_,_,_=fit_annular_templates(templates,observed,valid,r,np.linspace(.2,4.5,18),
                                          fixed_first=3.,upper_bounds=[np.inf,4.,np.inf])
        self.assertLessEqual(bounded[1],4.)
        self.assertAlmostEqual(bounded[0],3.)

    def test_missing_annuli_are_unknown_and_insufficient_data_rejected(self):
        image=np.ones((5,5))
        r=np.zeros_like(image)
        valid=np.ones_like(image,dtype=bool)
        self.assertTrue(np.isnan(annular_means(image,valid,r,[1,2])[0]))
        with self.assertRaises(ValueError):
            fit_annular_templates([image],image,valid,r,[0,1,2])


if __name__=='__main__':
    unittest.main()
