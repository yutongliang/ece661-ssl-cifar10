"""CPU regressions for experiment reliability; no downloads or real training runs."""
import argparse
import contextlib
import csv
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data
import linear_eval
import models
import pretrain_rotnet as rotnet
import train_classifier
import utils


class TinyCIFAR(torch.utils.data.Dataset):
    def __init__(self, train, transform, **kwargs):
        self.targets = [i % 10 for i in range(20 if train else 10)]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        pixels = np.random.RandomState(index).randint(0, 256, (32, 32, 3), dtype=np.uint8)
        return self.transform(Image.fromarray(pixels)), self.targets[index]


class InterruptedRun(Exception):
    pass


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(data, "SUBSET_DIR", str(self.root / "subsets")))
        self.stack.enter_context(patch.object(utils, "RESULTS_CSV", str(self.root / "results.csv")))
        for module in (data, rotnet, linear_eval):
            self.stack.enter_context(patch.object(module, "cifar10", TinyCIFAR))
        for module in (rotnet, linear_eval, train_classifier):
            self.stack.enter_context(patch.object(module, "get_device", lambda: torch.device("cpu")))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def run_script(self, module, args):
        with patch.object(sys, "argv", [module.__name__, *args]):
            module.main()

    def pretrain_args(self, path):
        return ["--epochs", "2", "--batch-size", "10", "--num-workers", "0",
                "--eval-every", "1", "--out", str(path)]

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_interrupted_resume_matches_uninterrupted_training(self):
        full = self.root / "full.pt"
        resumed = self.root / "resumed.pt"
        self.run_script(rotnet, self.pretrain_args(full))
        real_save = utils.save_checkpoint

        def disconnect_after_epoch(path, state):
            real_save(path, state)
            if state["epoch"] == 1:
                raise InterruptedRun()

        with patch.object(rotnet, "save_checkpoint", disconnect_after_epoch):
            with self.assertRaises(InterruptedRun):
                self.run_script(rotnet, self.pretrain_args(resumed))
        self.run_script(rotnet, self.pretrain_args(resumed) + ["--resume"])
        a = torch.load(str(full) + ".last", weights_only=True)
        b = torch.load(str(resumed) + ".last", weights_only=True)
        for key in ("encoder", "head", "optimizer", "scheduler", "scaler", "rng_state", "epoch"):
            self.assert_state_equal(a[key], b[key])

    def test_resume_rejects_missing_legacy_and_changed_settings(self):
        path = self.root / "resume.pt"
        args = self.pretrain_args(path) + ["--resume"]
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.run_script(rotnet, args)
        utils.save_checkpoint(str(path) + ".last", {"epoch": 1})
        with self.assertRaisesRegex(ValueError, "Legacy"):
            self.run_script(rotnet, args)
        self.run_script(rotnet, self.pretrain_args(path))
        for flag, value in [("--lr", "0.2"), ("--epochs", "3"), ("--seed", "4"),
                            ("--batch-size", "5"), ("--num-workers", "1")]:
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "settings differ"):
                self.run_script(rotnet, args + [flag, value])

    def test_rng_round_trip_after_weights_only_loading(self):
        utils.set_seed(42)
        path = self.root / "rng.pt"
        utils.save_checkpoint(path, utils.capture_rng_state(torch.device("cpu")))
        expected = (random.random(), np.random.rand(), torch.rand(3))
        utils.set_seed(99)
        utils.restore_rng_state(torch.load(path, weights_only=True), torch.device("cpu"))
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assert_state_equal(expected, actual)

    def test_atomic_save_preserves_previous_checkpoint_on_failure(self):
        path = self.root / "new" / "nested" / "model.pt"
        utils.save_checkpoint(path, {"epoch": 1})
        with patch.object(torch, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                utils.save_checkpoint(path, {"epoch": 2})
        self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_downstream_saves_and_logs_complete_settings(self):
        checkpoint = self.root / "encoder.pt"
        pretrain_config = {"lr": 0.1, "epochs": 100, "batch_size": 128, "seed": 0}
        utils.save_encoder(checkpoint, models.ResNet20(), "rotnet",
                           argparse.Namespace(**pretrain_config))
        common = ["--epochs", "1", "--batch-size", "10", "--num-workers", "0",
                  "--min-epoch-size", "0", "--subset-seed", "7", "--weight-decay", "0.001"]
        saved = self.root / "new" / "classifier.pt"
        self.run_script(train_classifier, common + ["--save", str(saved)])
        self.run_script(train_classifier, common + ["--init", str(checkpoint)])
        self.run_script(linear_eval, ["--checkpoint", str(checkpoint), "--epochs", "1",
                                     "--num-workers", "0", "--subset-seed", "7"])
        self.run_script(linear_eval, ["--random-init", "--epochs", "1", "--num-workers", "0"])
        self.assertIn("model", torch.load(saved, weights_only=True))
        with open(utils.RESULTS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["experiment"] for r in rows],
                         ["supervised", "finetune", "linear_eval_pool1", "linear_eval_pool1"])
        self.assertEqual(rows[1]["subset_seed"], "7")
        self.assertEqual(rows[1]["weight_decay"], "0.001")
        self.assertEqual(rows[1]["min_epoch_size"], "0")
        self.assertEqual(rows[2]["batch_size"], "256")
        self.assertEqual(json.loads(rows[1]["pretrain_config"]), pretrain_config)
        self.assertEqual(json.loads(rows[1]["config"])["subset_seed"], 7)
        self.assertTrue(all(r["torch_version"] and r["torchvision_version"] for r in rows))

    def test_old_results_are_preserved_when_columns_are_added(self):
        with open(utils.RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "test_acc"])
            writer.writeheader()
            writer.writerow({"method": "old_run", "test_acc": 0.5})
        utils.log_result(method="new_run", test_acc=0.6, subset_seed=7)
        with open(utils.RESULTS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["method"], "old_run")
        self.assertEqual(rows[0]["test_acc"], "0.5")
        self.assertEqual(rows[0]["subset_seed"], "")
        self.assertEqual(rows[1]["subset_seed"], "7")

    def test_non_cuda_autocast_does_not_call_torch_autocast(self):
        with patch.object(torch, "autocast", side_effect=AssertionError("unsupported device")):
            for name in ("cpu", "mps"):
                with utils.autocast_context(torch.device(name)):
                    pass

    def test_frozen_features_and_balanced_subsets(self):
        encoder = models.ResNet20()
        before = {k: v.clone() for k, v in encoder.state_dict().items()}
        features, _ = linear_eval.extract_features(
            encoder, TinyCIFAR(True, data.test_transform), 2, 2, torch.device("cpu"), 0)
        self.assertEqual(features.shape, (20, 128))
        self.assertFalse(features.requires_grad)
        self.assert_state_equal(before, encoder.state_dict())
        targets = np.repeat(np.arange(10), 5000)
        for fraction, count in [(0.01, 500), (0.1, 5000), (1.0, 50000)]:
            indices = data.label_subset_indices(targets, fraction)
            self.assertEqual(len(set(indices)), count)
            self.assertTrue(np.all(np.bincount(targets[indices]) == count // 10))
            self.assertEqual(indices, data.label_subset_indices(targets, fraction))


if __name__ == "__main__":
    unittest.main()
