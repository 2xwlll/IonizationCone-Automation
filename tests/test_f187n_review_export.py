import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.maintenance.export_f187n_review import export_review, verify_archive, digest


class ReviewExportTests(unittest.TestCase):
    def fixture(self, root):
        run, dataset = root/'run', root/'dataset'
        model = run/'training/model'
        model.mkdir(parents=True)
        dataset.mkdir()
        (dataset/'valid.npy').write_bytes(b'valid mask')
        (dataset/'plan.json').write_text('{}')
        rows = []
        for identifier, split, variant in (('positive', 'test', 'full'),
                                            ('negative', 'test', 'cone_free'),
                                            ('training', 'train', 'full')):
            row = {'id': identifier, 'split': split, 'variant': variant}
            for field in ('image', 'mask'):
                name = identifier+'_'+field+'.npy'
                (dataset/name).write_bytes((identifier+' '+field).encode())
                row[field], row[field+'_sha256'] = name, digest(dataset/name)
            rows.append(row)
        manifest = {'band': 'F187N', 'projection': 'circular_cone_envelope_v2',
                    'samples': rows, 'plan_sha256': digest(dataset/'plan.json'),
                    'valid_file': 'valid.npy', 'valid_sha256': digest(dataset/'valid.npy')}
        (dataset/'manifest.json').write_text(json.dumps(manifest))
        (model/'config.json').write_text(json.dumps({'dataset': str(dataset), 'variant': 'full',
            'epochs': 2, 'dataset_manifest_sha256': digest(dataset/'manifest.json')}))
        (model/'test_report.json').write_text(json.dumps({'samples': [{'id': 'positive'}, {'id': 'negative'}], 'count': 2}))
        (model/'selection.json').write_text(json.dumps({'best_epoch': 2}))
        (model/'epoch_001.pth').write_bytes(b'other checkpoint')
        (model/'epoch_002.pth').write_bytes(b'selected checkpoint')
        (run/'logs').mkdir()
        (run/'logs/train-1.log').write_bytes(b'original log\n')
        (run/'code').mkdir()
        (run/'code/model.py').write_text('# frozen model\n')
        (run/'snapshot.json').write_text(json.dumps({'sha256': {'code/model.py': digest(run/'code/model.py')}}))
        return run, dataset

    def test_export_includes_all_test_inputs_and_only_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, dataset = self.fixture(root)
            original = {p: digest(p) for tree in (run, dataset) for p in tree.rglob('*') if p.is_file()}
            output = root/'exports/review.tar.gz'
            result = export_review(run, output, include_checkpoint=True)
            self.assertEqual(result['test_images'], 2)
            manifest = verify_archive(output)
            with tarfile.open(str(output)) as archive:
                names = archive.getnames()
                self.assertIn('run/training/model/epoch_002.pth', names)
                self.assertNotIn('run/training/model/epoch_001.pth', names)
                self.assertIn('dataset/positive_image.npy', names)
                self.assertIn('dataset/negative_mask.npy', names)
                self.assertNotIn('dataset/training_image.npy', names)
                self.assertEqual(archive.extractfile('run/logs/train-1.log').read(), b'original log\n')
            self.assertTrue(manifest['dataset_is_test_subset'])
            self.assertEqual(Path(str(output)+'.sha256').read_text().split()[0], digest(output))
            self.assertTrue(all(digest(p) == value for p, value in original.items()))
            before = digest(output)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                export_review(run, output)
            self.assertEqual(digest(output), before)

    def test_default_omits_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, _ = self.fixture(root)
            output = root/'review.tar.gz'
            export_review(run, output)
            manifest = verify_archive(output)
            self.assertFalse(manifest['selected_checkpoint_included'])
            self.assertFalse(any(name.endswith('.pth') for name in manifest['files']))

    def test_fit_diagnostic_export_keeps_training_and_holdout_roles_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, dataset = self.fixture(root)
            selection = {'groups': ['training'], 'sample_ids_by_variant': {'full': ['training']}}
            frozen = json.loads((run/'snapshot.json').read_text())
            frozen['overfit_diagnostic'] = {'dataset': str(dataset),
                'dataset_manifest_sha256': digest(dataset/'manifest.json'),
                'dataset_plan_sha256': digest(dataset/'plan.json'), 'fit_selection': selection}
            (run/'snapshot.json').write_text(json.dumps(frozen))
            (run/'diagnostics/full').mkdir(parents=True)
            (run/'diagnostics/full/selected.pth').write_bytes(b'fit checkpoint')
            (run/'diagnostics/summary.json').write_text(json.dumps({'fit_selection': selection,
                'dataset_manifest_sha256': digest(dataset/'manifest.json'),
                'variants': {'full': {'selected_epoch': 10, 'samples': [{'id': 'training'}]}}}))
            output = root/'fit_review.tar.gz'
            result = export_review(run, output, include_checkpoint=True)
            manifest = verify_archive(output)
            self.assertEqual(result['training_images'], 1)
            self.assertEqual(manifest['test_ids'], [])
            self.assertEqual(manifest['fit_ids'], ['training'])
            self.assertIn('dataset/training_image.npy', manifest['files'])
            self.assertNotIn('dataset/positive_image.npy', manifest['files'])
            self.assertIn('run/diagnostics/full/selected.pth', manifest['files'])

    def test_changed_test_array_blocks_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, dataset = self.fixture(root)
            (dataset/'positive_image.npy').write_bytes(b'changed after evaluation')
            with self.assertRaisesRegex(ValueError, 'Changed source file'):
                export_review(run, root/'review.tar.gz')
            self.assertFalse((root/'review.tar.gz').exists())

    def test_dataset_escape_is_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, dataset = self.fixture(root)
            (root/'outside.npy').write_bytes(b'outside')
            manifest_path = dataset/'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['samples'][0].update(image='../outside.npy', image_sha256=digest(root/'outside.npy'))
            manifest_path.write_text(json.dumps(manifest))
            config_path = run/'training/model/config.json'
            config = json.loads(config_path.read_text())
            config['dataset_manifest_sha256'] = digest(manifest_path)
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'Unsafe review path'):
                export_review(run, root/'review.tar.gz')
            self.assertFalse((root/'review.tar.gz').exists())


if __name__ == '__main__':
    unittest.main()
