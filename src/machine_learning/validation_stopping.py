"""Explicit validation-only checkpoint selection and optional early stopping."""
import math


def balanced_mask_score(positive_dice, negative_area):
    """Equal weight to positive Dice and negative pixel specificity.

    This is a checkpoint-selection objective, not detection accuracy, a physical
    metric, or a combined image-complexity score. Both classes must be present.
    """
    if any(v is None or not math.isfinite(v) or not 0 <= v <= 1
           for v in (positive_dice, negative_area)):
        raise ValueError('Finite positive Dice and negative area in [0,1] required')
    return .5*(positive_dice+1-negative_area)


class ValidationStopping:
    def __init__(self, config):
        self.metric = config.get('selection_metric', 'val_loss')
        if self.metric not in ('val_loss', 'val_positive_dice', 'val_balanced_mask_score'):
            raise ValueError('Unsupported validation selection metric')
        self.mode = 'min' if self.metric == 'val_loss' else 'max'
        early = config.get('early_stopping', {})
        self.enabled = early.get('enabled', False)
        self.patience = early.get('patience', 12)
        self.min_delta = early.get('min_delta', 0.)
        self.min_epochs = early.get('min_epochs', 1)
        if (not isinstance(self.enabled, bool) or not isinstance(self.patience, int) or
                self.patience < 1 or not isinstance(self.min_epochs, int) or self.min_epochs < 1 or
                not math.isfinite(self.min_delta) or self.min_delta < 0):
            raise ValueError('Invalid validation early-stopping settings')
        self.best_value = None
        self.best_epoch = None
        self.anchor = None
        self.stale_epochs = 0
        self.should_stop = False
        self.last_epoch = 0

    def update(self, metrics, epoch):
        if epoch != self.last_epoch+1:
            raise ValueError('Validation stopping requires consecutive one-based epochs')
        value = metrics[self.metric]
        if value is None or not math.isfinite(value):
            raise ValueError('Nonfinite or missing validation selection value')
        sign = -1 if self.mode == 'min' else 1
        improved = self.best_value is None or sign*(value-self.best_value) > 0
        if improved:
            self.best_value, self.best_epoch = value, epoch
        # Keep the actual best checkpoint even if progress is below min_delta.
        significant = self.anchor is None or sign*(value-self.anchor) > self.min_delta
        if significant:
            self.anchor, self.stale_epochs = value, 0
        else:
            self.stale_epochs += 1
        self.last_epoch = epoch
        self.should_stop = self.enabled and epoch >= self.min_epochs and self.stale_epochs >= self.patience
        return improved

    def state(self):
        return {'metric': self.metric, 'mode': self.mode, 'enabled': self.enabled,
                'patience': self.patience, 'min_delta': self.min_delta, 'min_epochs': self.min_epochs,
                'best_value': self.best_value, 'best_epoch': self.best_epoch,
                'stale_epochs': self.stale_epochs, 'stop_triggered': self.should_stop,
                'epochs_completed': self.last_epoch}
