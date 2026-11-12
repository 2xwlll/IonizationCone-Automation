import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


PATH=Path(__file__).resolve().parents[1]/'scripts/maintenance/launch_f187n_pipeline.py'
spec=importlib.util.spec_from_file_location('f187n_launcher',PATH)
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


class F187NLauncherTests(unittest.TestCase):
    def prepare(self, root):
        source=root/'source';env=root/'env';run=root/'experiment'
        (env/'bin').mkdir(parents=True);(env/'bin/python').write_text('test')
        for name in launcher.INPUTS:
            path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'research input')
        for name in ('configs/cloudy/f187n_dataset_pilot_v3.json','configs/cloudy/train_f187n_pilot_v2.json'):
            path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
        for name in ('scripts/slurm/generate_f187n.slurm','scripts/slurm/train_f187n.slurm','src/model.py'):
            path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# source\n')
        with patch.object(launcher,'ROOT',source):launcher.snapshot(run,env,3,1,'full',False)
        return source,env,run

    def test_snapshot_copies_inputs_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            source,env,run=self.prepare(Path(folder))
            launcher.verify_snapshot(run)
            (source/launcher.INPUTS[0]).write_bytes(b'new external input')
            launcher.verify_snapshot(run)  # Frozen copy is independent.
            (run/'code/src/model.py').write_text('# changed\n')
            with self.assertRaises(ValueError):launcher.verify_snapshot(run)

    def test_missing_hii_completion_manifest_fails_before_creating_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,env,_=self.prepare(root)
            (source/'data/cloudy/hii_f187n_calculated_v2/manifest.json').unlink()
            fresh=root/'incomplete'
            with patch.object(launcher,'ROOT',source):
                with self.assertRaises(ValueError):launcher.snapshot(fresh,env,3,1,'full',False)
            self.assertFalse(fresh.exists())

    def test_dependency_submission_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,run=self.prepare(Path(folder))
            args=['launcher','--run-root',str(run),'--resume']
            with patch('sys.argv',args),patch.object(launcher.shutil,'which',return_value='/bin/sbatch'), \
                    patch.object(launcher.subprocess,'check_output',side_effect=['101;arc\n','102;arc\n']) as submit, \
                    contextlib.redirect_stdout(io.StringIO()):
                launcher.main()
            commands=[c.args[0] for c in submit.call_args_list]
            self.assertNotIn('--dependency=afterok:101',commands[0])
            self.assertIn('--dependency=afterok:101',commands[1])
            self.assertIn('--kill-on-invalid-dep=yes',commands[1])
            env=submit.call_args_list[0].kwargs['env']
            self.assertEqual(env['AGN_DATASET'],str(run/'dataset'))
            self.assertEqual(env['AGN_SMOKE_ONLY'],'0')
            with patch('sys.argv',args),patch.object(launcher.shutil,'which',return_value='/bin/sbatch'), \
                    patch.object(launcher.subprocess,'check_output',return_value='101\n') as check:
                with self.assertRaises(RuntimeError):launcher.main()
            self.assertEqual(check.call_count,1)  # Query only: no duplicate sbatch.
            self.assertFalse((run/'.submission.lock').exists())

    def test_explicit_training_configuration_keeps_cap_unless_overridden(self):
        for override in (None, 3):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as folder:
                root=Path(folder);source,env,_=self.prepare(root)
                config=root/'guarded.json'
                config.write_text(json.dumps({'epochs':150,'selection_metric':'val_balanced_mask_score',
                    'early_stopping':{'enabled':True,'patience':12}}))
                run=root/'new_guarded'
                args=['launcher','--run-root',str(run),'--env-prefix',str(env),
                      '--train-config',str(config),'--dry-run']
                if override is not None:args+=['--epochs',str(override)]
                with patch('sys.argv',args),patch.object(launcher,'ROOT',source),contextlib.redirect_stdout(io.StringIO()):
                    launcher.main()
                frozen=json.loads((run/'code/configs/cloudy/f187n_train_launch.json').read_text())
                self.assertEqual(frozen['epochs'],override or 150)
                self.assertTrue(frozen['early_stopping']['enabled'])
                self.assertEqual(frozen['selection_metric'],'val_balanced_mask_score')

    def test_failed_training_submission_preserves_generation_id(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,run=self.prepare(Path(folder))
            with patch('sys.argv',['launcher','--run-root',str(run),'--resume']), \
                    patch.object(launcher.shutil,'which',return_value='/bin/sbatch'), \
                    patch.object(launcher.subprocess,'check_output',side_effect=[
                        '201\n',subprocess.CalledProcessError(1,['sbatch'])]):
                with self.assertRaises(subprocess.CalledProcessError):launcher.main()
            state=json.loads(next((run/'submissions').glob('*.json')).read_text())
            self.assertEqual(state['status'],'submission_failed')
            self.assertEqual(state['generation_job'],'201')
            self.assertNotIn('training_job',state)


if __name__=='__main__':unittest.main()
