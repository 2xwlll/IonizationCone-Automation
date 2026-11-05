import unittest

import torch

from src.machine_learning.models.model_2d import UNet
from src.machine_learning.losses.f187n import F187NLoss
from scripts.cloudy.train_f187n_pilot import binary_scores


class F187NLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_loss_rewards_correct_positive_and_empty_predictions(self):
        target=torch.zeros(2,1,32,32);target[0,:,8:24,8:24]=1
        criterion=F187NLoss()
        good=((target*2-1)*8).requires_grad_()
        bad=-good.detach()
        self.assertLess(float(criterion(good,target).detach()),float(criterion(bad,target).detach()))
        criterion(good,target).backward()
        self.assertTrue(torch.isfinite(good.grad).all())
        # BCE gives substantial positive gradients to false positives on empty targets.
        logits=torch.full((1,1,32,32),8.,requires_grad=True)
        empty=torch.zeros_like(logits)
        bce,dice=criterion.components(logits,empty)
        self.assertEqual(float(dice.detach()),0.)
        criterion(logits,empty).backward()
        self.assertGreater(float(logits.grad.sum()),.99)
        self.assertTrue(torch.isfinite(bce))

    def test_extreme_logits_are_finite_for_mixed_and_empty_batches(self):
        for target_value in (0.,1.):
            logits=torch.tensor([-1000.,1000.]).reshape(2,1,1,1).requires_grad_()
            loss=F187NLoss()(logits,torch.full_like(logits,target_value))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_production_model_learns_fixed_mixed_batch(self):
        torch.manual_seed(42)
        model=UNet(in_channels=1,out_channels=1,features=[16,32,64,128])
        criterion=F187NLoss();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        # Spatial structure and an empty image; no generator-label circularity.
        target=torch.zeros(2,1,64,64);target[0,:,16:48,16:48]=1
        inputs=target*2+.05*torch.randn_like(target)
        first=None
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            logits=model(inputs)
            self.assertEqual(logits.shape,target.shape)
            loss=criterion(logits,target)
            if first is None:first=float(loss.detach())
            loss.backward()
            for p in model.parameters():self.assertTrue(torch.isfinite(p.grad).all())
            optimizer.step()
        self.assertLess(float(loss.detach()),first*.8)
        model.eval()
        with torch.no_grad():self.assertTrue(torch.isfinite(model(inputs)).all())

    def test_empty_scores_and_false_positive_area(self):
        import numpy as np
        aperture=np.ones((10,10),bool);empty=np.zeros_like(aperture)
        self.assertEqual(binary_scores(empty,empty,aperture)['iou'],1.)
        predicted=empty.copy();predicted[0,:]=True
        scores=binary_scores(predicted,empty,aperture)
        self.assertEqual(scores['iou'],0.)
        self.assertEqual(scores['predicted_area_fraction'],.1)

    def test_evaluation_separates_negatives_and_never_assigns_them_angles(self):
        import numpy as np
        from pathlib import Path
        import tempfile
        from torch.utils.data import DataLoader,TensorDataset
        from scripts.cloudy.train_f187n_pilot import test_report
        class EmptyModel(torch.nn.Module):
            def forward(self,x):return torch.full_like(x,-20.)
        x=torch.ones(2,1,256,256);y=torch.zeros_like(x);y[0,:,:,64:192]=1
        ds=TensorDataset(x,y,torch.arange(2))
        ds.rows=[dict(id=str(i),group='g',split='test',variant='full' if i==0 else 'cone_free',
                      geometry={'axis_pa_east_of_north_deg':30.,'projected_full_opening_deg':80.} if i==0 else None,
                      complexity={},extent_over_optical_psf_fwhm=None,softening=1.) for i in range(2)]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);np.save(root/'valid.npy',np.ones((256,256),bool))
            report,_=test_report(EmptyModel(),DataLoader(ds,batch_size=2),root,
                {'matrix_arcsec':[[-.03,0],[0,.03]],'pixel_scale_arcsec':.03,'validation_scope':'test'},
                'cpu',{'prediction_threshold':.5,'negative_area_flag_fraction':.01})
        self.assertEqual((report['positive_count'],report['negative_count']),(1,1))
        self.assertEqual(report['positive_mean_model_annulus_iou'],0.)
        self.assertEqual(report['negative_mean_model_false_positive_area_fraction'],0.)
        negative=report['samples'][1]
        self.assertIsNone(negative['target_geometry'])
        self.assertIsNone(negative['model_angle_errors']['axis_error_deg'])


if __name__=='__main__':unittest.main()
