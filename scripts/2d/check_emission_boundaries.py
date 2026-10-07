#!/usr/bin/env python3
"""Regression checks for artificial radial arcs; run from the repository root."""

import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.ndimage import label
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    with patch.object(sys, "argv", ["generate_realistic_emission.py"]):
        generator = runpy.run_path("scripts/2d/generate_realistic_emission.py")
    generator["validate_config"]()

    # Size has a measurable spatial meaning, independent of total brightness.
    illum = np.zeros((128, 128), dtype=np.float32)
    illum[64, 64] = 1
    settings = generator["sample_cloud_settings"](0)
    settings.update(count=1, sigma_pixels=[3., 3.], axis_ratio=[1., 1.],
                    cluster_fraction=0., irregularity=0., luminosity_scatter=0.)
    state = np.random.get_state()
    cloud, diffuse = generator["cloud_components"](128, illum, settings, 42)
    after_state = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(state, after_state))
    yy, xx = np.mgrid[:128, :128]
    variance = np.sum(cloud * (xx - 64)**2) / cloud.sum()
    assert np.isclose(variance, 9., rtol=1e-5), variance
    assert np.allclose(cloud, cloud.T)
    uniform = np.ones_like(illum)
    for field in (cloud, diffuse):
        normalized = generator["normalize_component"](field, uniform)
        assert np.isclose(normalized.mean(), 1., rtol=1e-5)
    print("Passed cloud size, circular shape, component normalization and RNG isolation.")

    # Move the center across the radius of one fixed pixel. A binary radial
    # gate jumps here even when the underlying random field is unchanged.
    for radius in (20, 30, 50):
        for name in ("gas_texture", "dust_clumps"):
            values = []
            for offset in (-1e-4, 1e-4):
                np.random.seed(123)
                kwargs = dict(center_x=64 + offset, center_y=64)
                if name == "gas_texture":
                    field = generator[name](
                        128, 0, 40, radius, 0.9, 2,
                        lobe_seed=123, **kwargs,
                    )
                else:
                    field = generator[name](
                        128, 0, 40, radius, 0.4, [5, 12], **kwargs,
                    )
                assert np.isfinite(field).all()
                assert field.min() >= 0 and field.max() <= 1
                values.append(float(field[64, 64 + radius]))
            jump = abs(values[1] - values[0])
            assert jump < 1e-4, (name, radius, jump)
            print(f"{name}, radius {radius}: boundary jump {jump:.3g}")

    positives = negatives = 0
    for seed in range(40):
        image, mask, params = generator["generate_sample"](seed)
        for array in (image, mask):
            assert array.shape == (generator["GRID"], generator["GRID"])
            assert array.dtype == np.float32 and np.isfinite(array).all()
            assert array.min() >= 0 and array.max() <= 1
        if params["is_negative"]:
            negatives += 1
            assert not mask.any()
        else:
            positives += 1
            assert mask.any()
            components, count = label(mask > 0.5, structure=np.ones((3, 3)))
            assert count == 1, (seed, count)
            cy = int(round(params["center_y"]))
            cx = int(round(params["center_x"]))
            assert components[cy, cx] == 1, seed
        repeat_image, repeat_mask, repeat_params = generator["generate_sample"](seed)
        assert np.array_equal(image, repeat_image)
        assert np.array_equal(mask, repeat_mask)
        assert params == repeat_params
    assert positives and negatives
    print(f"Passed: {positives} positive and {negatives} negative samples.")

    # Extreme settings exercise both visibility states without asserting an
    # astrophysical occurrence rate for blocked counter-cones.
    with patch.object(sys, "argv", ["generate_realistic_emission.py",
                                    "--obscured-counter-frac", "1"]):
        obscured = runpy.run_path("scripts/2d/generate_realistic_emission.py")
    obscured["mixture_config"]["bicone_fraction"] = 1.0
    generate_obscured = obscured["generate_sample"]
    seed = next(i for i in range(40) if not generate_obscured(i)[2]["is_negative"])
    _, hidden_mask, hidden = generate_obscured(seed)
    assert hidden["intrinsic_bicone"] and hidden["counter_lobe_obscured"]
    assert not hidden["bicone"]
    assert hidden["counter_transmission"] <= obscured["CONFIG"]["obscuration"]["counter_transmission"][1]
    assert label(hidden_mask > 0.5, structure=np.ones((3, 3)))[1] == 1
    generate_obscured.__globals__["OBSCURED_COUNTER_FRAC"] = 0.0
    _, visible_mask, visible = generate_obscured(seed)
    assert visible["intrinsic_bicone"] and visible["bicone"]
    assert not visible["counter_lobe_obscured"]
    assert label(visible_mask > 0.5, structure=np.ones((3, 3)))[1] == 1
    print("Passed visible and dust-obscured counter-cone states.")



if __name__ == "__main__":
    main()
