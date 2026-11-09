import unittest
import torch
from src.machine_learning.optimization import scaled_optimizer_step


class ScaledOptimizerTests(unittest.TestCase):
    def model(self):
        model=torch.nn.Linear(2,1)
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        return model,optimizer

    def backward(self,model,optimizer,scaler):
        optimizer.zero_grad(set_to_none=True)
        loss=model(torch.ones(2,2)).square().mean()
        self.assertTrue(torch.isfinite(loss))
        scaler.scale(loss).backward()

    def test_scaled_overflow_skips_update_and_recovers_at_lower_scale(self):
        model,optimizer=self.model()
        scaler=torch.amp.GradScaler('cpu',init_scale=16.,backoff_factor=.5)
        self.backward(model,optimizer,scaler)
        before=[p.detach().clone() for p in model.parameters()]
        next(model.parameters()).grad.fill_(float('inf'))
        result=scaled_optimizer_step(model,optimizer,scaler,5.)
        self.assertFalse(result['updated']);self.assertTrue(result['overflow_skipped'])
        self.assertEqual(result['scale_after'],8.)
        self.assertFalse(optimizer.state)
        for p,value in zip(model.parameters(),before):self.assertTrue(torch.equal(p,value))
        self.backward(model,optimizer,scaler)
        result=scaled_optimizer_step(model,optimizer,scaler,5.)
        self.assertTrue(result['updated'])
        self.assertTrue(any(not torch.equal(p,value) for p,value in zip(model.parameters(),before)))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_unscaled_nonfinite_gradient_stops_without_corrupting_weights(self):
        model,optimizer=self.model();scaler=torch.amp.GradScaler('cpu',enabled=False)
        self.backward(model,optimizer,scaler)
        before=[p.detach().clone() for p in model.parameters()]
        next(model.parameters()).grad.fill_(float('nan'))
        with self.assertRaisesRegex(RuntimeError,'without mixed-precision'):
            scaled_optimizer_step(model,optimizer,scaler,5.)
        for p,value in zip(model.parameters(),before):self.assertTrue(torch.equal(p,value))
        self.assertFalse(optimizer.state)

    def test_finite_large_gradients_are_clipped_before_update(self):
        model,optimizer=self.model();scaler=torch.amp.GradScaler('cpu',enabled=False)
        self.backward(model,optimizer,scaler)
        for p in model.parameters():p.grad.fill_(100.)
        result=scaled_optimizer_step(model,optimizer,scaler,1.)
        self.assertTrue(result['updated']);self.assertGreater(result['grad_norm'],1.)
        norm=torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))
        self.assertLessEqual(float(norm),1.00001)


if __name__=='__main__':unittest.main()
