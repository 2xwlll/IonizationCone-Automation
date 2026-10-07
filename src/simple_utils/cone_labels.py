"""Hard geometric labels for projected AGN ionization cones."""

import numpy as np
from scipy.ndimage import label


def connected_cone_mask(grid, params, cone_signal=None):
    """Return a hard binary cone or bicone defined only by geometry.

    ``cone_signal`` is accepted for compatibility and deliberately ignored.
    Gas brightness, dust, clouds, noise and PSF never alter the target.
    """
    yy, xx = np.mgrid[:grid, :grid]
    dx = xx - params["center_x"]
    dy = yy - params["center_y"]
    radius = np.hypot(dx, dy)
    axis = np.deg2rad(params["phi"])
    half_opening = np.deg2rad(params["opening"] * params["opening_scale"])
    mask = np.zeros((grid, grid), dtype=bool)

    for side in range(2 if params["bicone"] else 1):
        direction = axis + side * np.pi
        along = dx * np.cos(direction) + dy * np.sin(direction)
        across = -dx * np.sin(direction) + dy * np.cos(direction)
        angular_offset = np.abs(np.arctan2(across, np.maximum(along, 1e-8)))
        mask |= (
            (along >= 0)
            & (angular_offset <= half_opening)
            & (radius <= params["r_max"])
        )

    core_radius = max(1.5, min(float(params["nucleus_sigma"]), 3.0))
    mask |= radius <= core_radius

    _, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count != 1:
        raise RuntimeError(f"Geometric cone mask has {count} connected components")
    return mask.astype(np.float32)
