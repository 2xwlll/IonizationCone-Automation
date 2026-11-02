"""Exercise archive preservation while rejecting APIs unavailable on Python 3.6."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.maintenance import backup_removed_outputs as backup


class BackupTests(unittest.TestCase):
    def test_old_subprocess_api_and_symlink_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'results').mkdir()
            source = root/'results/example.png'
            source.write_bytes(b'original research bytes')
            link = root/'results/example.npy'
            link.symlink_to('../data/unavailable.npy')
            output = root/'backup'

            def git(command, **kwargs):
                self.assertNotIn('text', kwargs)
                if 'cwd' in kwargs:
                    self.assertIsInstance(kwargs['cwd'], str)
                if command[-1] == '--show-toplevel':
                    self.assertTrue(kwargs['universal_newlines'])
                    return str(root)+'\n'
                if command[:2] == ['git', 'diff']:
                    return b'results/example.png\0results/example.npy\0results/missing.png\0'
                self.assertTrue(kwargs['universal_newlines'])
                return 'example-head\n'

            # pathlib.Path.readlink was added after Python 3.6.
            with patch.object(backup.subprocess, 'check_output', side_effect=git), \
                 patch.object(Path, 'readlink', side_effect=AttributeError('not in Python 3.6'), create=True), \
                 patch('sys.argv', ['backup', '--output', str(output)]):
                backup.main()
            report = json.loads((output/'manifest.json').read_text())
            self.assertTrue(report['verified'])
            self.assertEqual(len(report['files']), 2)
            self.assertEqual(report['missing'], ['results/missing.png'])
            self.assertEqual(source.read_bytes(), b'original research bytes')
            self.assertTrue(link.is_symlink())
            self.assertEqual(report['files'][1]['link_target'], '../data/unavailable.npy')

    def test_no_removals_creates_no_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'backup'
            with patch.object(backup.subprocess, 'check_output', side_effect=[folder+'\n', b'']), \
                 patch('sys.argv', ['backup', '--output', str(output)]):
                backup.main()
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
