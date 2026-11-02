import unittest

import numpy as np

from scripts.cloudy.compare_real_synthetic_cones import (
    angle_error, excess_profile, mask_scores, measure_lobes, north_up, sky_offsets,
)


class ConeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.scale = .03
        self.matrix = np.array([[-.45, np.sqrt(1-.45**2)],
                                [np.sqrt(1-.45**2), .45]]) * self.scale
        self.center = [127.5, 127.5]

    def test_north_up_is_rotation_not_reflection_and_coordinates_roundtrip(self):
        transform = np.linalg.inv(self.matrix) @ np.diag([-self.scale, self.scale])
        self.assertAlmostEqual(np.linalg.det(transform), 1.)
        e, n = sky_offsets((256,256), self.center, self.matrix)
        data = 2*e + 3*n + 100
        reoriented = north_up(data, np.ones(data.shape,dtype=bool), self.center, self.matrix, self.scale)
        y, x = np.indices((257,257))
        expected = 2*(-(x-128)*self.scale) + 3*(y-128)*self.scale + 100
        good = np.isfinite(reoriented)
        self.assertTrue(np.allclose(reoriented[good], expected[good], atol=1e-10))

    def test_uniform_scene_does_not_create_a_cone(self):
        e,n = sky_offsets((256,256),self.center,self.matrix)
        image = np.ones(e.shape)*100
        p,_,_ = excess_profile(image,np.ones(e.shape,dtype=bool),e,n)
        self.assertTrue(np.allclose(p,0))
        self.assertIsNone(measure_lobes(p,.2)['primary'])

    def test_known_axis_recovered_and_overall_brightness_does_not_change_profile(self):
        e,n = sky_offsets((256,256),self.center,self.matrix)
        theta=np.rad2deg(np.arctan2(e,n))%360
        delta=abs((theta-30+180)%360-180)
        image=100*(1+np.exp(-.5*(delta/20)**2))*np.exp(-np.hypot(e,n)/4)
        valid=np.ones(e.shape,dtype=bool)
        p,_,_=excess_profile(image,valid,e,n)
        p2,_,_=excess_profile(image*7,valid,e,n)
        self.assertTrue(np.allclose(p,p2))
        self.assertAlmostEqual(measure_lobes(p,.2)['primary']['pa_east_of_north_deg'],30,delta=3)

    def test_mask_metrics_and_axis_periodicity(self):
        a=np.array([[True,True],[False,False]])
        b=np.array([[False,True],[True,False]])
        m=mask_scores(a,b,np.ones_like(a))
        self.assertAlmostEqual(m['iou'],1/3)
        self.assertAlmostEqual(m['dice'],.5)
        self.assertEqual(angle_error(210,30),0)
        self.assertEqual(angle_error(179,1),2)


if __name__=='__main__':unittest.main()
