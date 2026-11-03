from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:  # Local documentation runtime; GPU deployment has torch.
    torch = types.ModuleType("torch")
    sys.modules["torch"] = torch
try:
    import imageio.v3  # noqa: F401
except ImportError:
    imageio = types.ModuleType("imageio")
    imageio_v3 = types.ModuleType("imageio.v3")
    imageio.v3 = imageio_v3
    sys.modules["imageio"] = imageio
    sys.modules["imageio.v3"] = imageio_v3

from atas_geometry.memory.fusion import FusedSceneMemory, SimilarityTransform
from atas_geometry.register.matching import (
    AnchorSet,
    _fit_ransac,
    _collect_novel_frames,
    _lift_uv_robust,
    _pair_consensus,
    adaptive_overlap_radius,
    camera_prior_quality,
    classify_candidate_rays,
    fit_camera_prior,
    refine_rigid_icp,
    select_novel_points,
    transform_disagreement,
    validate_heldout,
    write_report,
)


def _transform(scale=1.2):
    angle = np.radians(12.0)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=np.float64)
    return SimilarityTransform(scale, rotation, np.asarray([0.4, -0.2, 0.1]))


def _memory(xyz, voxel=0.02):
    n = len(xyz)
    return FusedSceneMemory(
        xyz=np.asarray(xyz, dtype=np.float32), rgb=np.zeros((n, 3), dtype=np.uint8),
        world_time=np.zeros(n, dtype=np.float32), confidence=np.ones(n, dtype=np.float32),
        foreground=np.zeros(n, dtype=np.float32), weight=np.ones(n, dtype=np.float32),
        point_count=np.ones(n, dtype=np.int32), source_count=np.ones(n, dtype=np.int16),
        conflict_score=np.zeros(n, dtype=np.float32), provenance_offsets=np.arange(n + 1, dtype=np.int64),
        provenance_input_indices=np.arange(n, dtype=np.int32), voxel_size=voxel, time_bin_size=1.0,
        camera_ids=("home",), voxel_origin=np.zeros(3, dtype=np.float32),
    )


class FusionV2Tests(unittest.TestCase):
    def test_multiframe_ransac_recovers_sim3_with_outliers(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(240, 3))
        expected = _transform()
        target = expected.apply(source) + rng.normal(scale=0.002, size=source.shape)
        target[:50] = rng.normal(size=(50, 3)) * 8
        fitted, inliers = _fit_ransac(source, target, threshold=0.02)
        self.assertGreater(inliers.sum(), 175)
        self.assertAlmostEqual(fitted.scale, expected.scale, places=2)
        self.assertLess(transform_disagreement(fitted, expected)["rotation_deg"], 0.5)

    def test_camera_prior_uses_orientation_for_locked_camera(self):
        source = np.tile(np.eye(4), (5, 1, 1))
        wanted = _transform(scale=1.0)
        target = source.copy()
        target[:, :3, :3] = wanted.rotation
        target[:, :3, 3] = wanted.translation
        fitted = fit_camera_prior(source, target)
        self.assertIsNotNone(fitted)
        self.assertLess(transform_disagreement(fitted, wanted)["rotation_deg"], 0.1)

    def test_novelty_uses_l1_scale_not_old_16x_radius(self):
        existing = _memory([[0, 0, 0]], voxel=0.02)
        xyz = np.asarray([[0.01, 0, 0], [0.081, 0, 0], [0.085, 0, 0]], dtype=np.float32)
        rgb = np.arange(9, dtype=np.uint8).reshape(3, 3)
        novel_xyz, _, report = select_novel_points(xyz, rgb, existing, device=None)
        self.assertEqual(len(novel_xyz), 1)
        self.assertAlmostEqual(report["novelty_radius"], 0.025)
        self.assertGreater(float(novel_xyz[0, 0]), 0.07)

    def test_transform_disagreement_detects_bad_prior(self):
        a = _transform()
        b = SimilarityTransform(a.scale * 2, np.eye(3), a.translation + 3)
        report = transform_disagreement(a, b)
        self.assertGreater(report["rotation_deg"], 10)
        self.assertGreater(report["translation"], 4)

    def test_absurd_camera_prior_has_no_veto_power(self):
        source = np.tile(np.eye(4), (5, 1, 1))
        target = source.copy()
        source[:, 0, 3] = np.linspace(0, 2, 5)
        target[:, 0, 3] = np.linspace(0, 0.1, 5)
        absurd = SimilarityTransform(0.0557, np.eye(3), np.zeros(3))
        quality = camera_prior_quality(source, target, absurd)
        self.assertEqual(quality["status"], "prior_unusable")
        self.assertFalse(quality["veto_enabled"])
        self.assertTrue(quality["camera_follow_mismatch"])
        self.assertIn("prior_scale_out_of_range", quality["reasons"])

    def test_well_fitted_camera_prior_can_veto(self):
        source = np.tile(np.eye(4), (8, 1, 1))
        source[:, 0, 3] = np.linspace(0, 1, len(source))
        expected = _transform(scale=1.2)
        target = source.copy()
        target[:, :3, :3] = np.einsum("ij,fjk->fik", expected.rotation, source[:, :3, :3])
        target[:, :3, 3] = expected.apply(source[:, :3, 3])
        quality = camera_prior_quality(source, target, expected)
        self.assertEqual(quality["status"], "usable")
        self.assertTrue(quality["veto_enabled"])
        self.assertFalse(quality["camera_follow_mismatch"])

    def test_five_centimeter_heldout_error_is_not_precise_fusion(self):
        error = np.asarray([0.020, 0.035, 0.04796, 0.050, 0.080])
        result = validate_heldout(error, voxel_size=0.0048)
        self.assertFalse(result["accepted"])
        self.assertLess(result["median_limit"], result["median_error"])

    def test_overlap_radius_tracks_alignment_uncertainty(self):
        residual = np.asarray([0.020, 0.030, 0.04796, 0.050])
        radius = adaptive_overlap_radius(0.0048, residual)
        self.assertGreater(radius, 0.04)
        self.assertLessEqual(radius, 0.05)

    def test_precise_heldout_alignment_passes(self):
        error = np.asarray([0.008, 0.010, 0.012, 0.014, 0.020])
        result = validate_heldout(error, voxel_size=0.0048)
        self.assertTrue(result["accepted"])

    def test_icp_falls_back_when_same_split_does_not_improve(self):
        class FakeTree:
            def __init__(self, points):
                self.points = np.asarray(points)

            def query(self, query, k=1, workers=-1):
                del k, workers
                delta = np.asarray(query)[:, None, :] - self.points[None, :, :]
                squared = np.sum(delta * delta, axis=2)
                indices = np.argmin(squared, axis=1)
                return np.sqrt(squared[np.arange(len(query)), indices]), indices

        scipy = types.ModuleType("scipy")
        spatial = types.ModuleType("scipy.spatial")
        spatial.cKDTree = FakeTree
        scipy.spatial = spatial
        old_scipy = sys.modules.get("scipy")
        old_spatial = sys.modules.get("scipy.spatial")
        sys.modules["scipy"] = scipy
        sys.modules["scipy.spatial"] = spatial
        try:
            rng = np.random.default_rng(12)
            points = rng.normal(size=(300, 3))
            identity = SimilarityTransform(1.0, np.eye(3), np.zeros(3))
            chosen, report = refine_rigid_icp(points, points, identity, max_points=200)
        finally:
            if old_scipy is None:
                sys.modules.pop("scipy", None)
            else:
                sys.modules["scipy"] = old_scipy
            if old_spatial is None:
                sys.modules.pop("scipy.spatial", None)
            else:
                sys.modules["scipy.spatial"] = old_spatial
        self.assertFalse(report["accepted"])
        self.assertEqual(report["reason"], "heldout_geometry_not_improved")
        np.testing.assert_allclose(chosen.translation, identity.translation)

    def test_consistency_counts_frames_before_fine_voxel_dedup(self):
        existing = _memory([[0, 0, 0]], voxel=0.02)
        frame_xyz = np.asarray(
            [[1.0, 0.00, 0.0], [1.0, 0.05, 0.0], [1.0, 0.10, 0.0]],
            dtype=np.float32,
        )
        frame_rgb = np.full((3, 3), 100, dtype=np.uint8)
        identity = SimilarityTransform(1.0, np.eye(3), np.zeros(3))
        ray_report = {"candidate_points": 3, "outside_camera": 0, "unknown_ray_points": 3, "covered_overlap_points": 0, "covered_conflict_points": 0, "nearer_conflicts": 0, "farther_conflicts": 0}
        with patch("atas_geometry.register.matching._static_frame_points", return_value=(frame_xyz, frame_rgb)), patch(
            "atas_geometry.register.matching.classify_candidate_rays",
            return_value=(np.ones(3, dtype=bool), ray_report),
        ):
            recon = types.SimpleNamespace(
                cam_c2w=np.tile(np.eye(4), (2, 1, 1)),
                frames=[np.zeros((10, 10, 3), dtype=np.uint8)] * 2,
                intrinsics=np.tile(np.asarray([10, 10, 5, 5]), (2, 1)),
            )
            xyz, _, report = _collect_novel_frames(
                recon, [0, 1], identity, existing, device=None, overlap_radius=0.03,
                temporal_bin_count=2, min_temporal_bins=2,
            )
        self.assertEqual(report["multi_frame_support_cells"], 2)
        self.assertEqual(report["coherent_support_cells"], 2)
        self.assertEqual(len(xyz), 3)

    def test_ray_gate_rejects_covered_foreground_and_keeps_unknown(self):
        existing = np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32)
        candidate = np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 2.005], [0.5, 0.0, 2.0]],
            dtype=np.float32,
        )
        unknown, report = classify_candidate_rays(
            existing, candidate, np.eye(4), np.asarray([100, 100, 50, 50]),
            (100, 100), depth_tolerance=0.02, cell_pixels=2,
        )
        self.assertEqual(unknown.tolist(), [False, False, True])
        self.assertEqual(report["nearer_conflicts"], 1)
        self.assertEqual(report["covered_overlap_points"], 1)

    def test_three_bin_persistence_removes_middle_only_object(self):
        existing = _memory([[0, 0, 0]], voxel=0.02)
        wall = np.asarray([[1.0, 0.00, 0], [1.0, 0.05, 0], [1.0, 0.10, 0]], dtype=np.float32)
        rabbit = np.asarray([[2.0, 0.00, 0], [2.0, 0.05, 0], [2.0, 0.10, 0]], dtype=np.float32)
        rgb = np.full((6, 3), 100, dtype=np.uint8)

        def frame_points(_recon, index):
            points = np.concatenate((wall, rabbit)) if index in (3, 4, 5) else wall
            return points, rgb[: len(points)]

        def all_unknown(_existing, candidate, *_args, **_kwargs):
            count = len(candidate)
            return np.ones(count, dtype=bool), {"candidate_points": count, "outside_camera": 0, "unknown_ray_points": count, "covered_overlap_points": 0, "covered_conflict_points": 0, "nearer_conflicts": 0, "farther_conflicts": 0}

        recon = types.SimpleNamespace(
            cam_c2w=np.tile(np.eye(4), (9, 1, 1)),
            frames=[np.zeros((10, 10, 3), dtype=np.uint8)] * 9,
            intrinsics=np.tile(np.asarray([10, 10, 5, 5]), (9, 1)),
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "atas_geometry.register.matching._static_frame_points", side_effect=frame_points
        ), patch(
            "atas_geometry.register.matching.classify_candidate_rays", side_effect=all_unknown
        ):
            support_path = Path(tmp) / "support.npz"
            xyz, _, report = _collect_novel_frames(
                recon, range(9), SimilarityTransform(1, np.eye(3), np.zeros(3)),
                existing, device=None, overlap_radius=0.03,
                temporal_bin_count=3, min_temporal_bins=3,
                support_stats_path=support_path,
            )
            support = np.load(support_path)
        self.assertTrue(np.all(xyz[:, 0] < 1.5))
        self.assertGreaterEqual(report["candidate_cell_count"], 4)
        self.assertTrue(np.any(support["bin_support"] == 1))
        self.assertTrue(np.any(support["bin_support"] == 3))

    def test_robust_lift_rejects_depth_discontinuity(self):
        class Recon:
            intrinsics = np.asarray([[100.0, 100.0, 5.0, 5.0]])
            cam_c2w = np.tile(np.eye(4), (1, 1, 1))

            def depth_at(self, index):
                del index
                depth = np.ones((10, 10), dtype=np.float64)
                depth[4:7, 5:] = 2.0
                return depth

        _, valid, report = _lift_uv_robust(
            Recon(), 0, np.asarray([[2.2, 2.3], [4.8, 5.0]])
        )
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        self.assertEqual(report["depth_unstable"], 1)

    def test_pair_consensus_removes_inconsistent_scale_cluster(self):
        good_a = SimilarityTransform(1.0, np.eye(3), np.zeros(3))
        good_b = SimilarityTransform(1.02, np.eye(3), np.asarray([0.01, 0, 0]))
        bad = SimilarityTransform(1.7, _transform(scale=1.0).rotation, np.asarray([2, 0, 0]))
        reports = tuple(
            {
                "pair_id": index,
                "static_matches": 40,
                "pair_transform": transform.to_json_dict(),
            }
            for index, transform in enumerate((good_a, good_b, bad))
        )
        anchors = AnchorSet(
            source=np.arange(27, dtype=np.float64).reshape(9, 3),
            target=np.arange(27, dtype=np.float64).reshape(9, 3),
            pair_ids=np.repeat(np.arange(3), 3),
            pair_reports=reports,
        )
        selected = _pair_consensus(anchors)
        self.assertEqual(set(selected.pair_ids.tolist()), {0, 1})

    def test_report_serializes_numpy_without_writing_elsewhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            write_report({"array": np.asarray([1, 2]), "scalar": np.float32(3)}, path)
            self.assertTrue(path.is_file())
            self.assertIn('"array"', path.read_text())





if __name__ == "__main__":
    unittest.main()
