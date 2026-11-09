import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.maintenance import retry_f187n_training as retry
from scripts.maintenance import launch_f187n_pipeline as launch


class RetryTests(unittest.TestCase):
    def prepare_source(self,root):
        source=root/'original';code=source/'code';env=root/'env';env.mkdir()
        (env/'bin').mkdir();(env/'bin/python').write_text('placeholder')
        for name in launch.INPUTS:
            p=code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original frozen scientific input')
        launch_config=code/'configs/cloudy/f187n_train_launch.json'
        launch_config.parent.mkdir(parents=True);launch_config.write_text(json.dumps({'epochs':30,'variant':'full','seed':42}))
        (source/'submissions').mkdir()
        (source/'snapshot.json').write_text(json.dumps({'environment':str(env),'groups':64,
            'sha256':{str(p.relative_to(source)):launch.digest(p) for p in code.rglob('*') if p.is_file()}}))
        dataset=source/'dataset';dataset.mkdir()
        (dataset/'manifest.json').write_text(json.dumps({'band':'F187N','projection':'circular_cone_envelope_v2'}))
        (dataset/'plan.json').write_text('{}')
        previous=source/'training/model';previous.mkdir(parents=True)
        (previous/'config.json').write_text(json.dumps({'dataset_manifest_sha256':launch.digest(dataset/'manifest.json')}))
        return source,env

    def fake_snapshot(self,target,env,groups,epochs,variant,smoke,scientific_root=None,training_settings=None):
        self.assertEqual(training_settings['seed'],42)
        self.assertEqual(scientific_root,self.source/'code')
        target.mkdir();(target/'logs').mkdir();(target/'submissions').mkdir()
        settings={'sha256':{},'environment':str(env),'groups':groups,'variant':variant,'epochs':epochs}
        (target/'snapshot.json').write_text(json.dumps(settings))
        return settings

    def test_training_only_submission_reuses_and_pins_original_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.source,env=self.prepare_source(root);target=root/'retry'
            original=launch.digest(self.source/'snapshot.json')
            args=['retry','--source-run',str(self.source),'--run-root',str(target)]
            with patch('sys.argv',args),patch.object(retry,'snapshot',side_effect=self.fake_snapshot), \
                    patch.object(retry.shutil,'which',return_value='/bin/sbatch'), \
                    patch.object(retry.subprocess,'check_output',return_value='12345\n') as submit, \
                    contextlib.redirect_stdout(io.StringIO()):
                retry.main()
            self.assertEqual(submit.call_count,1)
            command=submit.call_args.args[0]
            self.assertTrue(command[-1].endswith('train_f187n.slurm'))
            self.assertFalse(any('dependency=' in s for s in command))
            environment=submit.call_args.kwargs['env']
            self.assertEqual(environment['AGN_DATASET'],str(self.source/'dataset'))
            self.assertEqual(environment['AGN_DATASET_MANIFEST_SHA256'],launch.digest(self.source/'dataset/manifest.json'))
            self.assertEqual(original,launch.digest(self.source/'snapshot.json'))
            self.assertFalse((target/'dataset').exists())

    def test_changed_failed_run_dataset_is_rejected_before_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,_=self.prepare_source(root)
            (source/'dataset/manifest.json').write_text(json.dumps({'band':'F187N','projection':'circular_cone_envelope_v2','changed':True}))
            with self.assertRaisesRegex(ValueError,'failed training attempt'):
                retry.prepare(source,root/'retry')
            self.assertFalse((root/'retry').exists())

    def test_source_active_jobs_block_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            source,_=self.prepare_source(Path(folder))
            (source/'submissions/attempt.json').write_text(json.dumps({'training_job':'987'}))
            with patch.object(retry.subprocess,'check_output',return_value='987\n'):
                with self.assertRaisesRegex(RuntimeError,'active jobs'):retry.check_inactive(source)


if __name__=='__main__':unittest.main()
