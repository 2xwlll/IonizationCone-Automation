"""Finite-gradient updates with explicit mixed-precision overflow accounting."""
import torch


def scaled_optimizer_step(model, optimizer, scaler, max_grad_norm):
    """Call after scaled loss.backward(); return whether an update occurred.

    GradScaler checks for nonfinite gradients in unscale_(). On FP16 overflow,
    let its public step/update API skip the update and lower the scale BEFORE
    attempting clipping. Clipping a nonfinite gradient would propagate NaNs;
    raising first prevents the scaler's normal recovery. Genuine FP32 failures
    and a nonfinite norm of otherwise finite gradients still fail explicitly.
    """
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    if not gradients or any(g is None for g in gradients):
        raise RuntimeError('Missing gradients in trainable model parameters')
    finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
    before = float(scaler.get_scale())
    if not finite:
        if not scaler.is_enabled():
            raise RuntimeError('Nonfinite gradients without mixed-precision scaling; optimizer not updated')
        # unscale_ recorded the overflow, so step skips parameter/optimizer updates.
        scaler.step(optimizer)
        scaler.update()
        after = float(scaler.get_scale())
        if after >= before:
            raise RuntimeError('Gradient overflow did not reduce the loss scale')
        return {'updated': False, 'overflow_skipped': True, 'grad_norm': None,
                'scale_before': before, 'scale_after': after}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return {'updated': True, 'overflow_skipped': False, 'grad_norm': float(norm),
            'scale_before': before, 'scale_after': float(scaler.get_scale())}
