import unittest

import numpy as np

from src.simple_utils.cone_geometry import CIRCULAR, INCLINATION, LEGACY, resolve_cone_geometry


def config(alpha=35., inclination=45., projection=CIRCULAR):
    return dict(opening_half_angle_deg=alpha, inclination_deg=inclination,
                position_angle_east_of_north_deg=30., projection=projection,
                inclination_convention=INCLINATION)


class ConeGeometryTests(unittest.TestCase):
    def test_projection_matches_independently_rotated_3d_boundary(self):
        # Construct a circular cone around +x, then rotate in 3D, then project.
        phi = np.linspace(0, 2*np.pi, 200001)
        for alpha, inc in [(10, 0), (40, 5), (35, 45), (35, -45), (5, 84), (70, 15)]:
            with self.subTest(alpha=alpha, inc=inc):
                a, i = np.deg2rad([alpha, inc])
                boundary = np.array([np.full_like(phi, np.cos(a)),
                                     np.sin(a)*np.cos(phi), np.sin(a)*np.sin(phi)])
                rotation = np.array([[np.cos(i), 0, -np.sin(i)], [0, 1, 0],
                                     [np.sin(i), 0, np.cos(i)]])
                projected = rotation @ boundary
                measured = np.ptp(np.rad2deg(np.arctan2(projected[1], projected[0])))
                expected = resolve_cone_geometry(config(alpha, inc))['projected_full_opening_deg']
                self.assertAlmostEqual(measured, expected, delta=1e-5)

    def test_zero_tilt_and_tilt_sign(self):
        self.assertAlmostEqual(resolve_cone_geometry(config(35, 0))['projected_full_opening_deg'], 70)
        self.assertEqual(resolve_cone_geometry(config(35, 45))['projected_full_opening_deg'],
                         resolve_cone_geometry(config(35, -45))['projected_full_opening_deg'])

    def test_legacy_config_retains_old_expression_exactly(self):
        for alpha, inc in [(35,45), (20,80), (40,-5)]:
            c = config(alpha, inc)
            del c['projection']
            del c['inclination_convention']
            result = resolve_cone_geometry(c)
            original = np.arctan(np.tan(np.deg2rad(alpha))/max(np.cos(np.deg2rad(inc)), .2))
            self.assertEqual(result['projected_half_opening_rad'], original)
            self.assertEqual(result['projection'], LEGACY)
            self.assertFalse(result['projection_explicit_in_config'])

    def test_invalid_or_ambiguous_geometry_rejected(self):
        cases = [config(0,0), config(90,0), config(35,90), config(35,-90),
                 config(35,55), config(35,-56), config(float('nan'),0),
                 config(35,float('inf')), config(projection='typo')]
        c = config(); c.pop('inclination_convention'); cases.append(c)
        c = config(); c['inclination_convention'] = 'from_line_of_sight'; cases.append(c)
        for c in cases:
            with self.subTest(config=c), self.assertRaises(ValueError):
                resolve_cone_geometry(c)


if __name__ == '__main__':
    unittest.main()
