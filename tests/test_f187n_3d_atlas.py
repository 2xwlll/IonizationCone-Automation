import unittest
import numpy as np
from scipy.ndimage import gaussian_filter
from scripts.cloudy.plot_f187n_3d_atlas import spatial_fields


class AtlasFieldTests(unittest.TestCase):
    def test_patchiness_field_average_recovers_the_defined_scalar(self):
        y,x=np.indices((24,24));cone=1+x*x/40.;other=2+y/12.
        aperture=(x>4)&(x<19)&(y>3)&(y<20)
        fields,summaries=spatial_fields(cone,other,np.ones_like(cone),aperture,.1,
                                       np.ones_like(cone),np.exp(-np.ones_like(cone)))
        for name,layer in [('cone_residual',cone),('other_residual',other)]:
            residual=abs(layer-gaussian_filter(layer,2.,mode='reflect',truncate=4.))
            expected=float(residual[aperture].sum()/layer[aperture].sum())
            self.assertAlmostEqual(summaries[name],expected,places=13)
            linear_field=np.power(10.,fields[name])-1
            self.assertAlmostEqual(float(linear_field[aperture].mean()),expected,places=13)

    def test_local_competing_fraction_is_not_the_summed_contrast(self):
        cone=np.array([[1.,100.],[1.,100.]])
        other=np.array([[1.,1.],[9.,9.]])
        fields,_=spatial_fields(cone,other,np.ones_like(cone),np.ones_like(cone,dtype=bool),
                                .1,np.zeros_like(cone),np.ones_like(cone))
        np.testing.assert_allclose(fields['fraction'],other/(cone+other))
        contrast=other.sum()/cone.sum()
        self.assertNotAlmostEqual(float(fields['fraction'].mean()),contrast)
        self.assertNotAlmostEqual(float(fields['fraction'].mean()),other.sum()/(cone+other).sum())

    def test_undefined_local_values_stay_undefined(self):
        cone=np.zeros((4,4));other=np.ones((4,4));other[0,0]=0
        noise=np.ones((4,4));noise[0,0]=0
        fields,_=spatial_fields(cone,other,noise,np.ones((4,4),bool),.1,
                                np.zeros((4,4)),np.ones((4,4)))
        self.assertTrue(np.isnan(fields['fraction'][0,0]))
        self.assertTrue(np.isnan(fields['signal_sigma'][0,0]))
        self.assertTrue(np.isnan(fields['cone_residual']).all())
        self.assertEqual(fields['fraction'][1,1],1.)
        self.assertEqual(fields['signal_sigma'][1,1],0.)


if __name__=='__main__':unittest.main()
