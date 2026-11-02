import unittest
import numpy as np
from src.simple_utils.gas_structure import soft_agn_emission_weight,nuclear_spatial_profile


class GasStructureTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'angular_transition_deg':10.,'boundary_variation_deg':7.,
                  'radial_transition_fraction':.1,'off_axis_fraction':.025,
                  'radial_decay':2.5,'central_gas_sigma_arcsec':.23,
                  'central_gas_amplitude':0.}

    def weight(self,angles,radii):
        r=np.asarray(radii,dtype=float);a=np.asarray(angles,dtype=float)
        return soft_agn_emission_weight(r*np.cos(a),r*np.sin(a),r,np.pi/4,
                                       .85,np.ones_like(r),np.zeros_like(r),3.94,self.cfg)

    def test_emission_has_no_jump_at_nominal_angular_or_radial_label_boundary(self):
        angular=self.weight([np.pi/4-1e-5,np.pi/4+1e-5],[.5,.5])
        radial=self.weight([0.,0.],[.85-1e-5,.85+1e-5])
        self.assertLess(abs(angular[0]/angular[1]-1),1e-3)
        self.assertLess(abs(radial[0]/radial[1]-1),1e-3)

    def test_on_axis_brighter_than_off_axis_and_emission_not_cut_off(self):
        values=self.weight([0.,np.pi/2,0.],[.5,.5,1.])
        self.assertTrue((values>0).all())
        self.assertGreater(values[0],values[1])
        self.assertGreater(values[0],values[2])

    def test_compact_nuclear_profile_conserves_flux_when_extent_changes(self):
        a=nuclear_spatial_profile((101,101),.01,.35,.10)
        b=nuclear_spatial_profile((101,101),.01,.35,.20)
        self.assertAlmostEqual(a.sum(),1.,places=12)
        self.assertAlmostEqual(b.sum(),1.,places=12)
        y,x=np.indices(a.shape);r2=(x-50)**2+(y-50)**2
        self.assertGreater(float((b*r2).sum()),float((a*r2).sum()))
        self.assertGreater(a[50,50],.65)


if __name__=='__main__':unittest.main()
