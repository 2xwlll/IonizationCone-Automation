import math
import unittest
from src.machine_learning.validation_stopping import ValidationStopping, balanced_mask_score


class ValidationStoppingTests(unittest.TestCase):
    def test_legacy_selects_minimum_loss_and_never_stops(self):
        control = ValidationStopping({})
        for epoch, value in enumerate((1., .5, .7, .8, .9), 1):
            control.update({'val_loss': value}, epoch)
        self.assertEqual(control.best_epoch, 2)
        self.assertFalse(control.should_stop)

    def test_stops_on_validation_plateau_and_keeps_actual_best(self):
        control = ValidationStopping({'selection_metric': 'val_positive_dice',
            'early_stopping': {'enabled': True, 'patience': 2, 'min_delta': .01}})
        for epoch, value in enumerate((.5, .6, .601, .602), 1):
            control.update({'val_positive_dice': value}, epoch)
        self.assertTrue(control.should_stop)
        self.assertEqual(control.best_epoch, 4)  # Small actual improvements remain selectable.
        self.assertAlmostEqual(control.best_value, .602)
        self.assertEqual(control.stale_epochs, 2)

    def test_real_progress_resets_patience_and_min_epochs_is_respected(self):
        control = ValidationStopping({'early_stopping': {'enabled': True, 'patience': 2, 'min_epochs': 5}})
        for epoch, value in enumerate((1., 1.1, 1.2, .9, 1., 1.1), 1):
            control.update({'val_loss': value}, epoch)
            if epoch < 6: self.assertFalse(control.should_stop)
        self.assertTrue(control.should_stop)
        self.assertEqual(control.best_epoch, 4)

    def test_empty_predictions_do_not_win_balanced_score(self):
        self.assertEqual(balanced_mask_score(0., 0.), .5)
        self.assertEqual(balanced_mask_score(1., 0.), 1.)
        self.assertGreater(balanced_mask_score(.6, .01), balanced_mask_score(.65, .4))
        for values in ((None, .1), (.5, math.nan), (1.1, 0.)):
            with self.assertRaises(ValueError): balanced_mask_score(*values)

    def test_invalid_control_and_nonfinite_metrics_fail(self):
        for config in ({'selection_metric': 'test_loss'}, {'early_stopping': {'patience': 0}},
                       {'early_stopping': {'min_delta': -1}}):
            with self.assertRaises(ValueError): ValidationStopping(config)
        control = ValidationStopping({})
        with self.assertRaises(ValueError): control.update({'val_loss': math.nan}, 1)
        with self.assertRaises(ValueError): control.update({'val_loss': 1.}, 2)


if __name__ == '__main__':
    unittest.main()
