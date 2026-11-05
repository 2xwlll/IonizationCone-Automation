import json
from pathlib import Path
import unittest

import numpy as np

from src.simple_utils.foreground_dust import patchy_screen,screen_summary
from src.simple_utils.image_descriptors import describe_image,set_sampled_parameter
from src.simple_utils.f187n import rebin_surface_brightness


class F187NDescriptorTests(unittest.TestCase):
    def config(self):
        return dict(model='patchy_screen_v1',tau_floor_pa_alpha=.1,tau_patch_pa_alpha=1.,
                    covering_fraction=.3,correlation_sigma_arcsec=.1)

    def test_screen_separates_depth_coverage_and_scale(self):
        cfg=self.config();tau,mask=patchy_screen((128,128),.03,51,cfg)
        repeat,repeat_mask=patchy_screen((128,128),.03,51,cfg)
        np.testing.assert_array_equal(tau,repeat);np.testing.assert_array_equal(mask,repeat_mask)
        self.assertEqual(mask.sum(),round(.3*mask.size))
        deeper,deep_mask=patchy_screen((128,128),.03,51,dict(cfg,tau_patch_pa_alpha=2.))
        np.testing.assert_array_equal(mask,deep_mask)
        self.assertTrue(np.all(np.exp(-deeper)<=np.exp(-tau)))
        _,large=patchy_screen((128,128),.03,51,dict(cfg,correlation_sigma_arcsec=.4))
        self.assertEqual(mask.sum(),large.sum());self.assertFalse(np.array_equal(mask,large))
        _,more=patchy_screen((128,128),.03,51,dict(cfg,covering_fraction=.6))
        self.assertTrue(np.all(more[mask]))

    def test_zero_full_and_invalid_screen_controls(self):
        for fraction in (0.,1.):
            tau,mask=patchy_screen((32,32),.03,1,dict(self.config(),covering_fraction=fraction))
            np.testing.assert_allclose(tau,.1+fraction)
            self.assertEqual(int(mask.sum()),int(fraction*mask.size))
        for key,value in [('covering_fraction',1.1),('tau_patch_pa_alpha',-1.),('correlation_sigma_arcsec',0.)]:
            with self.assertRaises(ValueError):patchy_screen((32,32),.03,1,dict(self.config(),**{key:value}))

    def test_transmission_is_averaged_after_exponentiation(self):
        tau=np.array([[0.,2.],[0.,2.]])
        mean_tau=rebin_surface_brightness(tau,2)
        mean_t=rebin_surface_brightness(np.exp(-tau),2)
        covered=rebin_surface_brightness((tau>0).astype(float),2)
        measured=screen_summary(mean_tau,mean_t,covered,np.ones((1,1),bool))
        self.assertAlmostEqual(measured['mean_transmission_pa_alpha'],(1+np.exp(-2))/2)
        self.assertNotAlmostEqual(measured['mean_transmission_pa_alpha'],np.exp(-1))
        self.assertEqual(measured['covered_area_fraction'],.5)

    def test_cone_local_measurement_and_categories_do_not_hide_off_cone_light(self):
        y,x=np.indices((64,64));mask=(abs(x-31.5)<8)&(abs(y-31.5)<20)
        valid=np.ones_like(mask);cone=np.ones(mask.shape);other=np.where(mask,.5,5.)
        maps=(np.full(mask.shape,.2),np.full(mask.shape,.7),np.full(mask.shape,.3))
        regions,dust,categories=describe_image(cone,other,mask,valid,.08,dust_maps=maps)
        self.assertAlmostEqual(regions['cone_annulus']['other_over_cone'],.5)
        self.assertGreater(regions['whole_annulus']['other_over_cone'],.5)
        self.assertEqual(categories['contrast'],'cone_brighter')
        self.assertEqual(categories['obscuration'],'moderate')
        empty=np.zeros_like(mask)
        regions,dust,categories=describe_image(np.zeros_like(cone),other,empty,valid,.08,dust_maps=maps)
        self.assertIsNone(regions['cone_annulus'])
        self.assertIsNone(regions['whole_annulus']['other_over_cone'])
        self.assertEqual(categories['contrast'],'cone_free')
        self.assertEqual(categories['obscuration_region'],'whole_annulus')
        _,dust,categories=describe_image(cone,np.zeros_like(other),mask,valid,.08,dust_maps=maps,dust_applied=False)
        self.assertFalse(dust['applied']);self.assertEqual(categories['obscuration'],'not_applied')

    def test_versioned_ranges_assign_only_existing_generator_controls(self):
        root=Path(__file__).resolve().parents[1]
        recipe=json.loads((root/'configs/cloudy/f187n_dataset_pilot_v3.json').read_text())
        cfg=json.loads((root/recipe['base_config']).read_text())
        geometry=dict(cfg['geometry'])
        for key,(low,high) in recipe['ranges'].items():set_sampled_parameter(cfg,key,(low+high)/2)
        self.assertEqual(cfg['foreground_dust']['covering_fraction'],.4)
        self.assertNotIn('messiness',recipe['ranges'])
        with self.assertRaises(KeyError):set_sampled_parameter(cfg,'foreground_dust.typo',1.)
        # Dust controls leave prescribed label geometry intact.
        current=dict(cfg['geometry']);set_sampled_parameter(cfg,'foreground_dust.tau_patch_pa_alpha',2.)
        self.assertEqual(current,cfg['geometry'])

    def test_report_preserves_undefined_negative_ratios_and_separate_dust_region(self):
        from scripts.cloudy.summarize_f187n_population import tabulate
        shape=(64,64);empty=np.zeros(shape,bool);valid=np.ones(shape,bool)
        regions,dust,categories=describe_image(np.zeros(shape),np.ones(shape),empty,valid,.08,
            dust_maps=(np.ones(shape),np.full(shape,.6),np.full(shape,.5)))
        row=tabulate(dict(id='negative',group='g',split='test',variant='cone_free',cone_present=False,
            complexity=regions['whole_annulus'],complexity_regions=regions,dust_diagnostics=dust,
            categories=categories,extent_over_optical_psf_fwhm=None))
        self.assertIsNone(row['M_cone_annulus']);self.assertIsNone(row['M_whole_annulus'])
        self.assertIsNone(row['cone_area_mean_transmission'])
        self.assertAlmostEqual(row['whole_area_mean_transmission'],.6)
        self.assertEqual(row['contrast_category'],'cone_free')


if __name__=='__main__':unittest.main()
