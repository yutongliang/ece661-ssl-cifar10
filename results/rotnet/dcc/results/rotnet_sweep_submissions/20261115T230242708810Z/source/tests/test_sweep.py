"""Grid integrity and Slurm submission checks; no network/Slurm/GPU required."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prioritize_rotnet
import rotnet_sweep
import submit_rotnet_sweep


class SweepTests(unittest.TestCase):
    def test_manifest_and_commands_cover_all_fifteen_without_finetuning(self):
        rows = rotnet_sweep.read_runs()
        self.assertEqual(len(rows), 15)
        self.assertEqual(len({row["run_id"] for row in rows}), 15)
        for row in rows:
            args = rotnet_sweep.training_arguments(row, Path("encoder.pt"))
            for flag, value in (("--epochs", "1000"), ("--checkpoint-every", "100"),
                                ("--lr", row["lr"]), ("--batch-size", row["global_batch_size"])):
                self.assertEqual(args[args.index(flag) + 1], value)
            self.assertIn("--require-cuda", args)
            self.assertNotIn("--init", args)
            self.assertNotIn("--fraction", args)

    def test_submission_freezes_source_and_gates_the_training_array(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            for name in submit_rotnet_sweep.SOURCE_FILES:
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
            with patch.object(submit_rotnet_sweep, "ROOT", project), \
                 patch.object(sys, "argv", ["submit_rotnet_sweep.py", "--dry-run"]), \
                 patch.object(submit_rotnet_sweep.subprocess, "run",
                              side_effect=AssertionError("dry-run must not call Slurm")), \
                 contextlib.redirect_stdout(io.StringIO()) as captured:
                submit_rotnet_sweep.main()
            output = captured.getvalue()
            self.assertIn("--array=0-14%2", output)
            self.assertIn("--dependency=afterok:900001", output)
            self.assertIn("--dependency=afterok:900002", output)
            self.assertIn("--partition=scavenger-gpu", output)
            self.assertIn("--gres=gpu:5000_ada:1", output)
            self.assertIn("--exclude=" + submit_rotnet_sweep.EXCLUDE, output)
            receipt_path = next(project.glob("results/rotnet_sweep_submissions/*/submission.json"))
            receipt = json.loads(receipt_path.read_text())
            source = Path(receipt["source"])
            self.assertEqual((source / "pretrain_rotnet.py").read_bytes(),
                             (ROOT / "pretrain_rotnet.py").read_bytes())
            self.assertEqual((source / "checkpoints").resolve(), (project / "checkpoints").resolve())
            self.assertEqual((source / "data").resolve(), (project / "data").resolve())
            self.assertEqual(set(receipt["jobs"]), {"preflight", "pilots", "train"})

    def test_priority_filter_excludes_running_jobs_other_users_and_supervisor(self):
        details = {"JobState": "PENDING", "UserId": "yl1064(123)",
                   "JobName": "cifar10_lrms3b_m5.example"}
        with patch.object(prioritize_rotnet.getpass, "getuser", return_value="yl1064"):
            self.assertTrue(prioritize_rotnet.eligible("12345", details["JobName"], details))
            self.assertFalse(prioritize_rotnet.eligible("12345", details["JobName"],
                                                      {**details, "JobState": "RUNNING"}))
            self.assertFalse(prioritize_rotnet.eligible("12345", details["JobName"],
                                                      {**details, "UserId": "someone(456)"}))
            self.assertFalse(prioritize_rotnet.eligible("12345", "lrm_seeds",
                                                      {**details, "JobName": "lrm_seeds"}))


if __name__ == "__main__":
    unittest.main()
