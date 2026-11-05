import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.simple_utils.f187n_pilot_dataset import (SCHEMA, LEGACY_SCHEMA, VARIANTS, ALL_VARIANTS,
    sha256, grouped_splits, normalize_image, validate_manifest, fit_mask_angles,
    cone_free_image, select_samples)


class PilotDatasetTests(unittest.TestCase):
    def test_group_split_reproducibility_and_counts(self):
        split=grouped_splits(64,731)
        self.assertEqual(split,grouped_splits(64,731))
        self.assertEqual([list(split.values()).count(s) for s in ('train','val','test')],[40,12,12])
        self.assertEqual(set(grouped_splits(3,731).values()),{'train','val','test'})
        with self.assertRaises(ValueError):grouped_splits(2,731)

    def test_invertible_normalization_preserves_signed_pixels(self):
        raw=np.linspace(-10,200,256).reshape(16,16)
        tensor,scale=normalize_image(raw)
        np.testing.assert_allclose(np.sinh(tensor.astype(float))*scale,raw,rtol=1e-6,atol=1e-5)
        with self.assertRaises(ValueError):normalize_image(np.zeros((8,8)))

    def test_mask_angle_fit_recovers_geometry_and_rejects_constant_masks(self):
        y,x=np.indices((256,256));east=-(x-127.5)*.03;north=(y-127.5)*.03
        angles=np.rad2deg(np.arctan2(east,north))%180
        valid=np.ones(x.shape,dtype=bool)
        for axis,width in [(30,90),(177,72),(90,108)]:
            mask=abs((angles-axis+90)%180-90)<=width/2
            fit=fit_mask_angles(mask,east,north,valid)
            self.assertLessEqual(abs((fit['axis_pa_mod180_deg']-axis+90)%180-90),3)
            self.assertLessEqual(abs(fit['projected_full_opening_deg']-width),3)
        self.assertIsNone(fit_mask_angles(np.zeros_like(valid),east,north,valid))
        self.assertIsNone(fit_mask_angles(valid,east,north,valid))

    def test_manifest_rejects_leakage_tampering_and_wrong_band(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);rows=[]
            for i,split in enumerate(('train','val','test')):
                for j,variant in enumerate(ALL_VARIANTS):
                    name=f'g{i}_{variant}';image=root/f'{name}.npy';mask=root/f'{name}_mask.npy'
                    np.save(image,np.full((256,256),i*4+j+1,dtype=np.float32))
                    present = variant != 'cone_free'
                    np.save(mask,np.full((256,256),int(present),dtype=np.uint8))
                    rows.append(dict(id=name,group=f'g{i}',seed=i,split=split,variant=variant,
                        image=image.name,mask=mask.name,image_sha256=sha256(image),mask_sha256=sha256(mask),softening=1.,
                        cone_present=present,geometry={} if present else None))
            np.save(root/'valid.npy',np.ones((256,256),dtype=bool))
            manifest=dict(schema=SCHEMA,band='F187N',shape=[1,256,256],projection='circular_cone_envelope_v2',
                          normalization='asinh(image/(0.03*p99(abs(image))))',samples=rows,
                          valid_sha256=sha256(root/'valid.npy'))
            def write(): (root/'manifest.json').write_text(json.dumps(manifest))
            write();validate_manifest(root)
            selected = select_samples(manifest, 'train', 'full')
            self.assertEqual({r['variant'] for r in selected}, {'full','cone_free'})
            negative=next(r for r in rows if r['variant']=='cone_free')
            negative['cone_present']=True;write()
            with self.assertRaises(ValueError):validate_manifest(root)
            negative['cone_present']=False
            # Legacy manifests stay readable without silently acquiring negatives.
            legacy=dict(manifest,schema=LEGACY_SCHEMA,samples=[r for r in rows if r['variant'] in VARIANTS])
            validate_manifest(root,manifest=legacy)
            rows[0]['split']='test';write()
            with self.assertRaises(ValueError):validate_manifest(root)
            rows[0]['split']='train';manifest['band']='F502N';write()
            with self.assertRaises(ValueError):validate_manifest(root)
            manifest['band']='F187N';write()
            (root/rows[0]['image']).write_bytes(b'tampered')
            with self.assertRaises(ValueError):validate_manifest(root)

    def test_counterfactual_removes_all_cone_light_and_preserves_other_noise(self):
        rng=np.random.default_rng(81)
        cone=rng.uniform(0,3,(12,12));other=rng.uniform(0,2,(12,12))
        noise=rng.normal(0,.1,(12,12));baseline=.7
        observed=cone+other+baseline+noise
        negative=cone_free_image(observed,cone)
        np.testing.assert_allclose(negative,other+baseline+noise,atol=1e-14)

    def test_completed_group_checks_geometry_and_source_before_skipping(self):
        from scripts.cloudy.prepare_f187n_dataset import completed_record
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'groups').mkdir()
            (root/'plan.json').write_text('{}');(root/'source.npz').write_bytes(b'source archive')
            group=dict(id='g0000',seed=1,split='train',geometry={'angle':42})
            rows=[]
            for variant in ALL_VARIANTS:
                present=variant!='cone_free';name='g0000_'+variant
                image=root/(name+'.npy');mask=root/(name+'_mask.npy')
                np.save(image,np.ones((2,2)));np.save(mask,np.full((2,2),present))
                rows.append(dict(id=name,group='g0000',seed=1,split='train',variant=variant,
                    cone_present=present,geometry=group['geometry'] if present else None,
                    image=image.name,mask=mask.name,image_sha256=sha256(image),mask_sha256=sha256(mask),
                    source_archive='source.npz',source_sha256=sha256(root/'source.npz')))
            path=root/'groups/g0000.json'
            def write():path.write_text(json.dumps({'plan_sha256':sha256(root/'plan.json'),'samples':rows}))
            write();completed_record(root,group)
            rows[0]['geometry']={'angle':99};write()
            with self.assertRaises(ValueError):completed_record(root,group)
            rows[0]['geometry']=group['geometry'];write()
            (root/'source.npz').write_bytes(b'changed raw components')
            with self.assertRaises(ValueError):completed_record(root,group)


if __name__=='__main__':unittest.main()
