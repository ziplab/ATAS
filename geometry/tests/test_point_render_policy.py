from __future__ import annotations

import unittest

from atas_geometry.render.point_render_policy import (
    bounded_projected_radius,
    resolve_point_render_mode,
)


class PointRenderPolicyTest(unittest.TestCase):
    def test_subpixel_footprint_stays_one_pixel(self) -> None:
        self.assertEqual(bounded_projected_radius(0.01, 1000.0, 10.0), 0)

    def test_two_pixel_footprint_uses_only_one_pixel_expansion(self) -> None:
        self.assertEqual(bounded_projected_radius(0.01, 1000.0, 5.0), 1)

    def test_near_point_never_becomes_a_large_disk(self) -> None:
        self.assertEqual(bounded_projected_radius(0.01, 1000.0, 0.5), 1)

    def test_zero_radius_switch_reproduces_single_pixel_policy(self) -> None:
        self.assertEqual(
            bounded_projected_radius(0.01, 1000.0, 0.5, max_radius=0), 0
        )

    def test_invalid_view_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bounded_projected_radius(0.01, 1000.0, 0.0)

    def test_render_mode_is_explicit_and_defaults_to_pixel(self) -> None:
        self.assertEqual(resolve_point_render_mode(None), "pixel")
        self.assertEqual(resolve_point_render_mode("bounded_square"), "bounded_square")
        with self.assertRaises(ValueError):
            resolve_point_render_mode("large_disks")


if __name__ == "__main__":
    unittest.main()
