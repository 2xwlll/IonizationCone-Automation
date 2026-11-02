"""Versioned angular envelopes for the 2D F187N pilot (not volume rendering)."""
import numpy as np


LEGACY = 'legacy_tangent_v1'
CIRCULAR = 'circular_cone_envelope_v2'
INCLINATION = 'from_sky_plane_positive_axis_receding'


def resolve_cone_geometry(geometry):
    """Return a projected half-angle and explicit interpretation.

    The positive projected axis points along PA east of north. Positive i tilts
    it away from the observer, consistent with the pilot's positive Doppler
    velocity. For v2, alpha is the intrinsic circular cone half-opening and
    sin(beta) = sin(alpha)/cos(i). Only sightlines strictly outside the cone are
    supported: the image/label recipe requires two separated angular sectors.
    A projected radial cutoff and empirical gas brightness are applied elsewhere.

    Missing projection selects the historical recipe solely for old configs.
    """
    alpha = float(geometry['opening_half_angle_deg'])
    inclination = float(geometry['inclination_deg'])
    pa = float(geometry['position_angle_east_of_north_deg'])
    if not np.isfinite([alpha, inclination, pa]).all():
        raise ValueError('Cone angles must be finite')
    if not 0 < alpha < 90 or not -90 < inclination < 90:
        raise ValueError('Require 0 < half-opening < 90 and -90 < inclination < 90 degrees')
    projection = geometry.get('projection', LEGACY)
    if projection == CIRCULAR:
        if geometry.get('inclination_convention') != INCLINATION:
            raise ValueError(f'Explicit inclination_convention must be {INCLINATION}')
        if alpha + abs(inclination) >= 90:
            raise ValueError('Sightline on/inside cone: separated-sector 2D labels are unsupported')
        half = np.arcsin(np.clip(np.sin(np.deg2rad(alpha)) /
                               np.cos(np.deg2rad(inclination)), 0., 1.))
        interpretation = 'exact angular envelope of an orthographically projected circular cone'
    elif projection == LEGACY:
        half = np.arctan(np.tan(np.deg2rad(alpha)) /
                         max(np.cos(np.deg2rad(inclination)), .2))
        interpretation = 'historical 2D approximation; not an exact circular 3D cone projection'
    else:
        raise ValueError(f'Unknown cone projection: {projection}')
    return {
        'projection': projection,
        'projection_explicit_in_config': 'projection' in geometry,
        'interpretation': interpretation,
        'input_opening_half_deg': alpha,
        'inclination_deg': inclination,
        'inclination_convention': INCLINATION,
        'inclination_convention_status': 'explicit' if projection == CIRCULAR else 'inferred from historical velocity prescription',
        'axis_pa_east_of_north_deg': pa % 360,
        'projected_half_opening_rad': float(half),
        'projected_half_opening_deg': float(np.rad2deg(half)),
        'projected_full_opening_deg': float(2*np.rad2deg(half)),
        'extent_definition': 'projected circular radial cutoff; not intrinsic cone axial length',
        'velocity_definition': 'empirical axial LOS velocity proportional to sin(i); no cone-wall velocity integration',
        'label_definition': 'nominal projected envelope with radial cutoff and connected central pixels; not detectable emission',
    }
