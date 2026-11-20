"""Grid integrity and Slurm submission checks; no network/Slurm/GPU required."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "slurm"))
import rotnet_sweep
import submit_rotnet_sweep
import shared_lock


class SweepTests(unittest.TestCase):
    def test_directory_lock_rejects_live_owner_and_releases_normally(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lock.d"
            with shared_lock.directory_lock(path):
                original = (path / "owner.json").read_text()
                with self.assertRaisesRegex(RuntimeError, "Lock is busy"):
                    with shared_lock.directory_lock(path):
                        self.fail("A second owner acquired a live lock")
                self.assertEqual((path / "owner.json").read_text(), original)
            self.assertFalse(path.exists())
            with shared_lock.directory_lock(path):
                self.assertTrue(path.is_dir())

    def test_directory_lock_recovers_a_crashed_process_without_flock(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lock.d"
            command = ("import os, sys; sys.path.insert(0, sys.argv[1]); "
                       "from shared_lock import directory_lock; "
                       "context = directory_lock(sys.argv[2]); context.__enter__(); os._exit(0)")
            subprocess.run([sys.executable, "-c", command, str(ROOT / "scripts" / "slurm"), str(path)], check=True)
            self.assertTrue(path.is_dir())
            with shared_lock.directory_lock(path):
                self.assertEqual(shared_lock.read_owner(path)["pid"], os.getpid())
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_directory_lock_recovers_previous_slurm_restart_but_not_unknown_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lock.d"
            path.mkdir()
            owner = {"host": "another-node", "pid": 123, "token": "old",
                     "slurm_job_id": "12345", "restart": 0}
            (path / "owner.json").write_text(json.dumps(owner))
            with patch.dict(os.environ, {"SLURM_JOB_ID": "12345", "SLURM_RESTART_COUNT": "1"}):
                with shared_lock.directory_lock(path):
                    self.assertEqual(shared_lock.read_owner(path)["restart"], 1)
            path.mkdir()
            (path / "owner.json").write_text(json.dumps({**owner, "slurm_job_id": ""}))
            with self.assertRaisesRegex(RuntimeError, "Lock is busy"):
                with shared_lock.directory_lock(path):
                    self.fail("An unknown remote owner must not be evicted")

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
            self.assertIn("--partition=gpu-common", output)
            self.assertIn("--gres=gpu:5000_ada:1", output)
            self.assertIn("--exclude=" + submit_rotnet_sweep.EXCLUDE, output)
            receipt_path = next(project.glob("results/rotnet_sweep_submissions/*/submission.json"))
            receipt = json.loads(receipt_path.read_text())
            source = Path(receipt["source"])
            for name in rotnet_sweep.TRAINING_SOURCE:
                self.assertEqual((source / name).read_bytes(), (ROOT / name).read_bytes())
            self.assertEqual((source / "checkpoints").resolve(), (project / "checkpoints").resolve())
            self.assertEqual((source / "data").resolve(), (project / "data").resolve())
            self.assertEqual(set(receipt["jobs"]), {"preflight", "pilots", "train"})


if __name__ == "__main__":
    unittest.main()
