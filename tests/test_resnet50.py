"""CPU checks for the ResNet-50 plan: encoder, accumulation, and downstream recipe."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cifar_ssl import data, downstream, models, utils
from cifar_ssl import pretrain_rotnet as rotnet
from test_training import TinyCIFAR


class Interrupted(Exception):
    pass


class FlatEncoder(nn.Module):
    """Batch-norm-free encoder, so accumulated and full-batch gradients must agree."""
    feature_dim = 8

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 8))

    def forward(self, x):
        return self.net(x)


class ResNet50PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(data, "SUBSET_DIR", str(self.root / "subsets")))
        stack.enter_context(patch.object(data, "cifar10", TinyCIFAR))
        stack.enter_context(patch.object(downstream, "get_device", lambda: torch.device("cpu")))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def test_resnet50_uses_the_b9_cifar_stem(self):
        encoder = models.build_encoder("resnet50")
        conv = encoder.stem[0]
        self.assertEqual((conv.kernel_size, conv.stride), ((3, 3), (1, 1)))
        self.assertFalse(any(isinstance(m, nn.MaxPool2d) for m in encoder.modules()))
        x = torch.randn(2, 3, 32, 32)
        self.assertEqual(encoder(x).shape, (2, 2048))
        shapes = [tuple(t.shape[1:]) for t in encoder(x, return_stages=True)]
        self.assertEqual(shapes, [(256, 32, 32), (512, 16, 16), (1024, 8, 8), (2048, 4, 4)])
        params = sum(p.numel() for p in encoder.parameters())
        self.assertTrue(23_000_000 < params < 24_000_000, params)

    def test_load_encoder_uses_the_recorded_architecture(self):
        new, old = self.root / "r50.pt", self.root / "r20.pt"
        utils.save_encoder(str(new), models.build_encoder("resnet50"), "rotnet",
                           argparse.Namespace(arch="resnet50"))
        utils.save_encoder(str(old), models.ResNet20(), "rotnet", argparse.Namespace(lr=0.1))
        self.assertIsInstance(models.load_encoder(str(new))[0], models.ResNet50CIFAR)
        self.assertIsInstance(models.load_encoder(str(old))[0], models.ResNet20)

    def test_accumulated_gradient_matches_the_full_batch(self):
        images = torch.randn(8, 3, 32, 32)
        grads = []
        for accum in (1, 4):
            torch.manual_seed(0)
            encoder, head = FlatEncoder(), nn.Linear(8, 4)
            params = list(encoder.parameters()) + list(head.parameters())
            optimizer = torch.optim.SGD(params, lr=0.0)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            _, _, count = rotnet.rotation_step(encoder, head, nn.CrossEntropyLoss(), images,
                                               optimizer, scaler, torch.device("cpu"), accum)
            self.assertEqual(count, 32)
            grads.append([p.grad.clone() for p in params])
        for full, accumulated in zip(*grads):
            self.assertTrue(torch.allclose(full, accumulated, atol=1e-6))
        with self.assertRaisesRegex(ValueError, "divisible"):
            rotnet.rotation_step(encoder, head, nn.CrossEntropyLoss(), images[:6], optimizer,
                                 scaler, torch.device("cpu"), 4)

    def test_plateau_rule_drops_once_then_stops(self):
        rule = downstream.PlateauRule(patience=2, threshold=0.01, max_drops=1)
        decisions = [rule.step(x) for x in [1.0, 0.5, 0.499, 0.498, 0.2, 0.199, 0.1985]]
        self.assertEqual(decisions, ["continue", "continue", "continue", "drop",
                                     "continue", "continue", "stop"])

    def make_inputs(self):
        encoder_path = self.root / "encoder.pt"
        utils.save_encoder(str(encoder_path), models.ResNet20(), "rotnet",
                           argparse.Namespace(arch="resnet20"))
        data.label_subset_indices(TinyCIFAR(True, None).targets, 0.5, 0)
        return encoder_path

    def args(self, out, *extra):
        return ["--fraction", "0.5", "--batch-size", "5", "--lr", "0.05", "--max-epochs", "4",
                "--plateau-patience", "1", "--min-epoch-size", "10", "--num-workers", "0",
                "--out-dir", str(out), *extra]

    def test_linear_eval_trains_only_the_classifier(self):
        encoder_path = self.make_inputs()
        out = self.root / "linear"
        downstream.main(self.args(out, "--init", str(encoder_path), "--freeze-encoder",
                                  "--keep-last"))
        result = json.loads((out / "result.json").read_text())
        self.assertEqual((result["experiment"], result["method"]), ("linear_eval", "rotnet"))
        before = torch.load(encoder_path, weights_only=True)["encoder"]
        after = torch.load(out / "last.pt", weights_only=True)["model"]
        for key, value in before.items():
            self.assertTrue(torch.equal(value, after["encoder." + key]), key)

    def test_interrupted_run_resumes_to_the_same_result_and_reruns_are_skipped(self):
        encoder_path = self.make_inputs()
        full, resumed = self.root / "full", self.root / "resumed"
        downstream.main(self.args(full, "--init", str(encoder_path)))
        real_save = downstream.save_checkpoint

        def crash_after_epoch_two(path, state):
            real_save(path, state)
            if state["epoch"] == 2:
                raise Interrupted()

        with patch.object(downstream, "save_checkpoint", crash_after_epoch_two):
            with self.assertRaises(Interrupted):
                downstream.main(self.args(resumed, "--init", str(encoder_path)))
        downstream.main(self.args(resumed, "--init", str(encoder_path)))
        a = json.loads((full / "result.json").read_text())
        b = json.loads((resumed / "result.json").read_text())
        for key in ("test_top1", "stop_epoch", "stop_reason", "lr_drops"):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual([h["loss"] for h in a["history"]], [h["loss"] for h in b["history"]])
        self.assertFalse((full / "last.pt").exists())
        downstream.main(self.args(full, "--init", str(encoder_path)))  # skipped, no error
        with self.assertRaisesRegex(ValueError, "different run"):
            downstream.main(self.args(full, "--init", str(encoder_path), "--lr", "0.1"))

    def test_supervised_requires_the_shared_subset(self):
        with self.assertRaisesRegex(FileNotFoundError, "shared label subset"):
            downstream.main(self.args(self.root / "sup", "--arch", "resnet20"))


if __name__ == "__main__":
    unittest.main()
