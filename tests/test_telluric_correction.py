"""Numerical checks of the correction operator, not a scientific calibration."""
import unittest
from pathlib import Path
import numpy as np
from src.simple_utils.telluric_correction import AtmosphereGrid,calibration_masks,corrected_with_errors


class TelluricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=Path(__file__).resolve().parents[1]/'data/spectra/ngc1068_skycalc_dry_v2'
        if not (folder/'manifest.json').exists():raise unittest.SkipTest('Downloaded ESO model fixture unavailable')
        cls.grid=AtmosphereGrid(folder)

    def test_independent_forward_and_continuum_inverse(self):
        g=self.grid;p=np.array([1.,np.log(6000.),.03,.2,.1,-.02]);w=np.arange(1760,2010,.06)
        x=(g.wave-1880)/100;c=np.exp(p[3]+p[4]*x+p[5]*x*x)
        forward=g.sample_source(c,p,w,atmosphere=True)
        truth=g.sample_source(c,p,w,atmosphere=False)
        prediction,_,t=g.evaluate(p,w)
        np.testing.assert_allclose(forward,prediction,rtol=1e-12)
        np.testing.assert_allclose(forward/t,truth,rtol=1e-12)

    def test_errors_grow_when_photons_lost(self):
        corrected,stat,total=corrected_with_errors(np.array([10.,10.]),np.array([1.,1.]),np.array([1.,.25]),.1)
        np.testing.assert_allclose(corrected,[10,40]);np.testing.assert_allclose(stat,[1,4])
        self.assertTrue(np.all(total>=stat))
        with self.assertRaises(ValueError):corrected_with_errors(1,1,np.array([0]),.1)

    def test_in_band_lines_never_fit_atmosphere(self):
        w=np.arange(1750,2015,.06)
        train,validation=calibration_masks(w,np.ones(w.shape,bool),np.ones(w.shape))
        self.assertFalse(np.any(train&validation))
        self.assertFalse(np.any((train|validation)&(w>=1858)&(w<=1905)))
        self.assertGreater(validation.sum(),20)

    def test_narrow_line_division_is_not_exact(self):
        g=self.grid;p=np.array([1.,np.log(6000.),0.,0.,0.,0.]);w=np.arange(1880,1885,.06)
        source=1+5*np.exp(-.5*((g.wave-1882.7)/.25)**2)
        measured=g.sample_source(source,p,w,atmosphere=True)
        truth=g.sample_source(source,p,w,atmosphere=False)
        teff=g.evaluate(p,w)[2]
        self.assertGreater(np.max(abs(measured/teff/truth-1)),.01)


if __name__=='__main__':unittest.main()
