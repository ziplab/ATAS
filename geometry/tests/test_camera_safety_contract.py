from __future__ import annotations

import unittest
from unittest.mock import patch
import sys
import types

import numpy as np

# The camera contract itself does not use image IO.  The avwm package
# imports the fusion module for a type used elsewhere, so keep this CPU-only
# unit test runnable in minimal developer environments as well as on the GPU
# service host.
try:
    import imageio.v3  # noqa: F401
except ModuleNotFoundError:
    imageio_stub = types.ModuleType("imageio")
    imageio_v3_stub = types.ModuleType("imageio.v3")
    imageio_stub.v3 = imageio_v3_stub
    sys.modules["imageio"] = imageio_stub
    sys.modules["imageio.v3"] = imageio_v3_stub

from atas_geometry.memory.camera_query import (
    CameraQueryRejected,
    resolve_camera_safety_contract,
    resolve_checked_camera_query,
)


def _precheck(passed: bool) -> dict[str, object]:
    return {
        "passed": passed,
        "failures": [] if passed else ["unsafe"],
        "camera_travel": 1.0,
    }


class CameraSafetyContractTest(unittest.TestCase):
    def test_missing_fields_preserve_legacy_substitution(self) -> None:
        self.assertEqual(
            resolve_camera_safety_contract({}),
            ("legacy_locked_fallback", True),
        )

    def test_same_semantic_policy_forbids_substitution(self) -> None:
        self.assertEqual(
            resolve_camera_safety_contract(
                {
                    "camera_safety_policy": "shrink_or_reject_same_semantic",
                    "allow_camera_semantic_substitution": True,
                }
            ),
            ("shrink_or_reject_same_semantic", False),
        )

    def test_explicit_false_also_forbids_substitution(self) -> None:
        self.assertEqual(
            resolve_camera_safety_contract(
                {"allow_camera_semantic_substitution": "false"}
            ),
            ("legacy_locked_fallback", False),
        )

    def test_same_semantic_contract_rejects_without_building_locked(self) -> None:
        scales: list[float] = []

        def build(scale: float) -> object:
            scales.append(scale)
            return object()

        with patch(
            "atas_geometry.memory.camera_query.check_camera_query",
            return_value=_precheck(False),
        ):
            with self.assertRaises(CameraQueryRejected) as caught:
                resolve_checked_camera_query(
                    build,
                    np.empty((0, 3)),
                    allow_semantic_substitution=False,
                )

        self.assertEqual(scales, [1.0, 0.5, 0.25])
        report = caught.exception.report
        self.assertTrue(report["rejected"])
        self.assertFalse(report["degraded_to_locked"])
        self.assertEqual(report["shrink_attempts"], 2)
        self.assertEqual(report["motion_scale"], 0.25)
        self.assertIn("same-semantic", str(report["reason"]))

    def test_same_semantic_contract_accepts_a_shrunk_named_move(self) -> None:
        scales: list[float] = []

        def build(scale: float) -> object:
            scales.append(scale)
            return {"scale": scale}

        with patch(
            "atas_geometry.memory.camera_query.check_camera_query",
            side_effect=[_precheck(False), _precheck(False), _precheck(True)],
        ):
            query, report = resolve_checked_camera_query(
                build,
                np.empty((0, 3)),
                allow_semantic_substitution=False,
            )

        self.assertEqual(scales, [1.0, 0.5, 0.25])
        self.assertEqual(query, {"scale": 0.25})
        self.assertFalse(report["rejected"])
        self.assertFalse(report["degraded_to_locked"])
        self.assertEqual(report["shrink_attempts"], 2)

    def test_legacy_contract_still_builds_locked_fallback(self) -> None:
        scales: list[float] = []

        def build(scale: float) -> object:
            scales.append(scale)
            return {"scale": scale}

        with patch(
            "atas_geometry.memory.camera_query.check_camera_query",
            return_value=_precheck(False),
        ):
            query, report = resolve_checked_camera_query(build, np.empty((0, 3)))

        self.assertEqual(scales, [1.0, 0.5, 0.25, 0.0])
        self.assertEqual(query, {"scale": 0.0})
        self.assertTrue(report["degraded_to_locked"])
        self.assertFalse(report["rejected"])


if __name__ == "__main__":
    unittest.main()
