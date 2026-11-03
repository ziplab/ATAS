from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from atas.runner.recon_ledger import (
    completed_result,
    ensure_recon_writeback,
    record_failed,
    record_intent,
    record_submitted,
    record_succeeded,
    resumable_job_id,
    result_is_complete,
)


def complete_result(job_id: str = "recon-1") -> dict[str, object]:
    return {
        "status": "succeeded",
        "job_id": job_id,
        "kind": "recon",
        "cameras_npz": "/gpu/run/story/shots/shot_01/l0/cameras.npz",
        "l0_dir": "/gpu/run/story/shots/shot_01/l0",
        "l1_npz": "/gpu/run/story/scenes/room/l1.npz",
        "l1_point_count": 125,
        "num_frames": 16,
        "fuse": {"merged": True, "novel_points": 25, "align": {"method": "sift"}},
    }


class ReconLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.shot_dir = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_intent_alone_never_counts_as_completed(self) -> None:
        record_intent(self.shot_dir, {"type": "recon", "video": "result.mp4"})
        self.assertIsNone(completed_result(self.shot_dir))
        self.assertIsNone(resumable_job_id(self.shot_dir))

    def test_interrupted_wait_resumes_same_job(self) -> None:
        record_submitted(self.shot_dir, job_id="recon-pending")
        self.assertEqual(resumable_job_id(self.shot_dir), "recon-pending")
        self.assertIsNone(completed_result(self.shot_dir))

    def test_failed_attempt_is_retryable(self) -> None:
        record_submitted(self.shot_dir, job_id="recon-bad")
        record_failed(self.shot_dir, error=RuntimeError("geometry HTTP 502"), job_id="recon-bad")
        self.assertIsNone(resumable_job_id(self.shot_dir))
        self.assertIsNone(completed_result(self.shot_dir))
        state = json.loads((self.shot_dir / "geometry_recon_state.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["attempt"], 1)

    def test_success_persists_full_writeback_and_skips_resume(self) -> None:
        record_submitted(self.shot_dir, job_id="recon-good")
        summary = record_succeeded(self.shot_dir, complete_result("recon-good"))
        self.assertEqual(summary["l1"]["point_count_before"], 100)
        self.assertEqual(summary["l1"]["point_count_after"], 125)
        self.assertEqual(summary["l1"]["version"], "recon-good")
        self.assertEqual(summary["l0"]["num_frames"], 16)
        self.assertEqual(summary["fusion"]["status"], "merged")
        self.assertEqual(summary["fusion"]["report"]["align"]["method"], "sift")
        self.assertIsNotNone(completed_result(self.shot_dir))
        self.assertIsNone(resumable_job_id(self.shot_dir))

    def test_incomplete_success_is_not_complete(self) -> None:
        result = complete_result()
        del result["cameras_npz"]
        self.assertFalse(result_is_complete(result))
        with self.assertRaisesRegex(RuntimeError, "complete writeback"):
            record_succeeded(self.shot_dir, result)

    def test_result_survives_crash_before_state_commit(self) -> None:
        # record_succeeded writes result before state. A complete result is the
        # recovery source even if state still says waiting.
        record_submitted(self.shot_dir, job_id="recon-crash-window")
        result = complete_result("recon-crash-window")
        (self.shot_dir / "geometry_recon_result.json").write_text(json.dumps(result))
        self.assertEqual(completed_result(self.shot_dir), result)

    def test_execution_resumes_interrupted_job_without_resubmitting(self) -> None:
        record_submitted(self.shot_dir, job_id="recon-existing")
        submitted: list[bool] = []
        waited: list[str] = []
        result, _, resumed = ensure_recon_writeback(
            self.shot_dir,
            submit=lambda: submitted.append(True) or "recon-new",
            wait=lambda job_id: waited.append(job_id) or complete_result(job_id),
        )
        self.assertTrue(resumed)
        self.assertEqual(submitted, [])
        self.assertEqual(waited, ["recon-existing"])
        self.assertEqual(result["job_id"], "recon-existing")

    def test_execution_retries_after_failure_and_records_second_attempt(self) -> None:
        record_submitted(self.shot_dir, job_id="recon-first")
        record_failed(self.shot_dir, error=RuntimeError("failed"), job_id="recon-first")
        result, _, resumed = ensure_recon_writeback(
            self.shot_dir,
            submit=lambda: "recon-second",
            wait=lambda job_id: complete_result(job_id),
        )
        self.assertFalse(resumed)
        self.assertEqual(result["job_id"], "recon-second")
        state = json.loads((self.shot_dir / "geometry_recon_state.json").read_text())
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["attempt"], 2)

    def test_execution_records_terminal_wait_failure(self) -> None:
        def fail(_: str) -> dict[str, object]:
            raise TimeoutError("wait interrupted")

        with self.assertRaisesRegex(TimeoutError, "wait interrupted"):
            ensure_recon_writeback(
                self.shot_dir,
                submit=lambda: "recon-timeout",
                wait=fail,
            )
        state = json.loads((self.shot_dir / "geometry_recon_state.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["job_id"], "recon-timeout")

    def test_uncertain_502_wait_preserves_job_for_resume(self) -> None:
        class Uncertain502(RuntimeError):
            recon_job_may_still_be_running = True

        with self.assertRaises(Uncertain502):
            ensure_recon_writeback(
                self.shot_dir,
                submit=lambda: "recon-possibly-running",
                wait=lambda _: (_ for _ in ()).throw(Uncertain502("geometry HTTP 502")),
            )
        self.assertEqual(resumable_job_id(self.shot_dir), "recon-possibly-running")
        state = json.loads((self.shot_dir / "geometry_recon_state.json").read_text())
        self.assertEqual(state["status"], "waiting")
        self.assertIn("502", state["last_poll_error"])

    def test_execution_skips_submit_and_wait_only_for_complete_success(self) -> None:
        record_succeeded(self.shot_dir, complete_result("recon-durable"))
        called: list[str] = []
        result, _, resumed = ensure_recon_writeback(
            self.shot_dir,
            submit=lambda: called.append("submit") or "unexpected",
            wait=lambda _: called.append("wait") or {},
        )
        self.assertTrue(resumed)
        self.assertEqual(result["job_id"], "recon-durable")
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
