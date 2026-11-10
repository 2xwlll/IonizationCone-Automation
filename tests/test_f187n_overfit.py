import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from scripts.maintenance import launch_f187n_overfit as launcher
from scripts.maintenance.launch_f187n_pipeline import INPUTS, digest
from scripts.cloudy.overfit_f187n import boundary_f1, recovery_checks


def sample_manifest():
    rows = []
    for split in ('train', 'val', 'test'):
        for group in range(6):
            name = split+str(group)
            for variant in ('full', 'smooth_control', 'clumpy_cone', 'cone_free'):
                rows.append({'id': name+'_'+variant, 'group': name, 'split': split, 'variant': variant})
    return {'band': 'F187N', 'projection': 'circular_cone_envelope_v2', 'samples': rows}


class OverfitDiagnosticTests(unittest.TestCase):
    def test_same_training_groups_used_for_all_representations(self):
        manifest = sample_manifest()
        variants = ['smooth_control', 'clumpy_cone', 'full']
        result = launcher.select_fit_samples(manifest, 4, 42, variants)
        self.assertEqual(result, launcher.select_fit_samples(manifest, 4, 42, variants))
        self.assertEqual(len(result['groups']), 4)
        self.assertTrue(all(g.startswith('train') for g in result['groups']))
        for variant, ids in result['sample_ids_by_variant'].items():
            self.assertEqual(len(ids), 8)
            self.assertEqual(ids[::2], [g+'_'+variant for g in result['groups']])
            self.assertEqual(ids[1::2], [g+'_cone_free' for g in result['groups']])

    def test_incomplete_controls_are_rejected_instead_of_using_holdouts(self):
        manifest = sample_manifest()
        manifest['samples'] = [r for r in manifest['samples'] if r['id'] != 'train0_smooth_control']
        with self.assertRaisesRegex(ValueError, 'Missing training control'):
            launcher.select_fit_samples(manifest, 6, 42, ['smooth_control'])
        with self.assertRaisesRegex(ValueError, 'Not enough training'):
            launcher.select_fit_samples(manifest, 7, 42, ['full'])

    def test_boundary_matching_penalizes_shift_and_empty_prediction(self):
        target = np.zeros((32, 32), bool); target[8:24, 8:24] = True
        shifted = np.roll(target, 1, axis=1)
        self.assertEqual(boundary_f1(target, target, 0), 1.)
        self.assertLess(boundary_f1(shifted, target, 0), 1.)
        self.assertEqual(boundary_f1(shifted, target, 2), 1.)
        self.assertEqual(boundary_f1(np.zeros_like(target), target, 2), 0.)

    def test_good_average_cannot_hide_failed_individual_or_negative(self):
        criteria = {'mean_positive_dice_at_least': .95, 'minimum_positive_dice_at_least': .9,
                    'mean_positive_boundary_f1_at_least': .9, 'maximum_negative_area_at_most': .01}
        good = {'mean_positive_dice': .98, 'minimum_positive_dice': .96,
                'mean_positive_boundary_f1': .95, 'maximum_negative_area': .001}
        self.assertTrue(recovery_checks(good, criteria))
        self.assertFalse(recovery_checks(dict(good, minimum_positive_dice=.5), criteria))
        self.assertFalse(recovery_checks(dict(good, maximum_negative_area=.1), criteria))

    def test_launcher_submits_only_one_gpu_diagnostic_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'source'; dataset = root/'dataset'; env = root/'env'
            (env/'bin').mkdir(parents=True); (env/'bin/python').write_text('placeholder')
            dataset.mkdir(); (dataset/'manifest.json').write_text(json.dumps(sample_manifest()))
            (dataset/'plan.json').write_text('{}')
            for name in INPUTS:
                path = source/'code'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('original input')
            cfg = source/'code/configs/cloudy/f187n_train_launch.json'; cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({'epochs': 30, 'variant': 'full', 'seed': 42}))
            (source/'training/model').mkdir(parents=True)
            (source/'training/model/config.json').write_text(json.dumps({'dataset': str(dataset),
                'dataset_manifest_sha256': digest(dataset/'manifest.json')}))
            (source/'submissions').mkdir()
            (source/'snapshot.json').write_text(json.dumps({'environment': str(env), 'groups': 18,
                'sha256': {str(p.relative_to(source)): digest(p) for p in (source/'code').rglob('*') if p.is_file()}}))
            before = digest(source/'snapshot.json'); target = root/'diagnostic'
            argv = ['launcher', '--source-run', str(source), '--run-root', str(target)]
            with patch('sys.argv', argv), patch.object(launcher.shutil, 'which', return_value='/bin/sbatch'), \
                    patch.object(launcher.subprocess, 'check_output', return_value='123\n') as submit, \
                    contextlib.redirect_stdout(io.StringIO()):
                launcher.main()
            self.assertEqual(submit.call_count, 1)
            self.assertTrue(submit.call_args.args[0][-1].endswith('overfit_f187n.slurm'))
            self.assertEqual(submit.call_args.kwargs['env']['AGN_SNAPSHOT_ROOT'], str(target))
            self.assertEqual(digest(source/'snapshot.json'), before)
            self.assertFalse((target/'dataset').exists())
            with patch('sys.argv', argv), patch.object(launcher.shutil, 'which', return_value='/bin/sbatch'), \
                    patch.object(launcher.subprocess, 'check_output') as repeated:
                with self.assertRaises(FileExistsError): launcher.main()
                repeated.assert_not_called()


if __name__ == '__main__':
    unittest.main()
