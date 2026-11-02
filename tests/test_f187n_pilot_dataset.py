import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.simple_utils.f187n_pilot_dataset import (SCHEMA, VARIANTS, sha256, grouped_splits,
    normalize_image, validate_manifest, fit_mask_angles)


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
                for j,variant in enumerate(VARIANTS):
                    name=f'g{i}_{variant}';image=root/f'{name}.npy';mask=root/f'{name}_mask.npy'
                    np.save(image,np.full((256,256),i*3+j+1,dtype=np.float32))
                    np.save(mask,np.ones((256,256),dtype=np.uint8))
                    rows.append(dict(id=name,group=f'g{i}',seed=i,split=split,variant=variant,
                        image=image.name,mask=mask.name,image_sha256=sha256(image),mask_sha256=sha256(mask),softening=1.))
            manifest=dict(schema=SCHEMA,band='F187N',shape=[1,256,256],projection='circular_cone_envelope_v2',
                          normalization='asinh(image/(0.03*p99(abs(image))))',samples=rows)
            def write(): (root/'manifest.json').write_text(json.dumps(manifest))
            write();validate_manifest(root)
            rows[0]['split']='test';write()
            with self.assertRaises(ValueError):validate_manifest(root)
            rows[0]['split']='train';manifest['band']='F502N';write()
            with self.assertRaises(ValueError):validate_manifest(root)
            manifest['band']='F187N';write()
            (root/rows[0]['image']).write_bytes(b'tampered')
            with self.assertRaises(ValueError):validate_manifest(root)


if __name__=='__main__':unittest.main()
