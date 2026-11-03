from __future__ import annotations

import unittest

from atas.world.geometry_client import compile_camera_intent, render_payload
def payload(move: str, *, breath_index: int, degrees: float = 0.0, camera_path=()):
    return render_payload(
        run_id="run",
        story_id="story",
        shot_id="shot",
        scene_id="scene",
        event_id="event",
        duration=4.0,
        fps=24,
        camera_move_name=move,
        camera_side_degrees=12.0,
        camera_movement_degrees=degrees,
        start_from_shot_id="shot_01",
        start_from_role="previous_shot",
        scene_revisit=False,
        geometry_layers="l1",
        camera_path=camera_path,
        breath_index=breath_index,
        geometry_shots_in_scene=8,
    )


class DirectorCameraContractTests(unittest.TestCase):
    def test_explicit_moves_are_not_replaced_by_shot_index_cycle(self) -> None:
        cases = (
            ("push_in", "push_in", 0.0),
            ("orbit", "orbit", 11.0),
            ("pull_back", "pull_back", 0.0),
            ("locked", "locked", 0.0),
        )
        for requested, expected, degrees in cases:
            with self.subTest(move=requested):
                outputs = [payload(requested, breath_index=index, degrees=degrees) for index in range(8)]
                self.assertEqual({item["camera_move_name"] for item in outputs}, {expected})
                self.assertEqual({item["camera_movement_degrees"] for item in outputs}, {degrees})
                self.assertTrue(all(item["camera_semantic_preserved"] for item in outputs))
                self.assertTrue(all(not item["camera_intent_fallback_used"] for item in outputs))
                self.assertTrue(
                    all(item["camera_intent_source"] == "screenplay_camera_move_name" for item in outputs)
                )

    def test_no_intent_uses_audited_legacy_fallback(self) -> None:
        output = payload("", breath_index=2)
        self.assertEqual(output["camera_move_name"], "push_in")
        self.assertTrue(output["camera_intent_fallback_used"])
        self.assertFalse(output["camera_semantic_preserved"])
        self.assertEqual(output["camera_intent_source"], "legacy_continue_cycle_fallback")
        self.assertIn("no camera movement", output["camera_intent_fallback_reason"])

    def test_unsupported_explicit_move_is_rejected_not_substituted(self) -> None:
        with self.assertRaisesRegex(ValueError, "refusing to replace"):
            payload("truck", breath_index=1)

    def test_path_can_supply_missing_dolly_semantic(self) -> None:
        path = (
            {"position": [0.0, -3.0, 1.5], "look_at": [0.0, 0.0, 1.5]},
            {"position": [0.0, -2.0, 1.5], "look_at": [0.0, 0.0, 1.5]},
        )
        output = payload("", breath_index=3, camera_path=path)
        self.assertEqual(output["camera_move_name"], "push_in")
        self.assertEqual(output["camera_intent_source"], "screenplay_camera_path")
        self.assertFalse(output["camera_intent_fallback_used"])

    def test_locked_scale_is_zero(self) -> None:
        plan = compile_camera_intent(
            camera_move_name="fixed",
            camera_movement_degrees=20,
        )
        self.assertEqual(plan["camera_move_name"], "locked")
        self.assertEqual(plan["camera_movement_degrees"], 0.0)
        self.assertEqual(plan["named_move_scale"], 0.0)


if __name__ == "__main__":
    unittest.main()
