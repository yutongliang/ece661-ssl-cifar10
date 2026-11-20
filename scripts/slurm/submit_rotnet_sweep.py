#!/usr/bin/env python3
"""Submit preflight -> resumable pilot -> 15-model RotNet array on DCC."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from rotnet_sweep import MANIFEST, TRAINING_SOURCE, atomic_json, read_runs
from shared_lock import directory_lock

ROOT = Path(__file__).resolve().parents[2]
EXCLUDE = ("dcc-gehmlab-gpu-ferc-s-z25-18,dcc-carlsonlab-gpu-ferc-s-h36-24,"
           "dcc-core-gpu-ferc-s-h36-6,dcc-majoroslab-gpu-ferc-a-r32-34")
SWEEP_JOB = "scripts/slurm/rotnet_sweep.sbatch"
SOURCE_FILES = (
    *TRAINING_SOURCE, "requirements.txt",
    "cifar_ssl/linear_eval.py", "cifar_ssl/train_classifier.py",
    "tests/test_training.py", "tests/test_sweep.py", SWEEP_JOB,
    "scripts/slurm/rotnet_sweep.py", "scripts/slurm/shared_lock.py",
    "scripts/slurm/submit_rotnet_sweep.py",
    str(MANIFEST.relative_to(ROOT)),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", default="gpu-common")
    parser.add_argument("--gpu-type", default="5000_ada", help="32 GB initial pilot request")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--python", default=os.environ.get("ECE661_PYTHON",
                        str(Path.home() / "miniforge3/envs/ece661-project4/bin/python")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    read_runs()
    if not 1 <= args.concurrency <= 15:
        parser.error("concurrency must be between 1 and 15")
    if not args.dry_run and not Path(args.python).is_file():
        parser.error(f"Environment Python is missing: {args.python}; pass --python explicitly")
    if not args.dry_run:
        queue = subprocess.run(["squeue", "--array", "-h", "-u", os.environ["USER"], "-o", "%i"],
                               check=True, capture_output=True, text=True).stdout.splitlines()
        if len(queue) + 17 > 400:
            raise RuntimeError("Need room for 17 jobs (2 gates + 15 tasks) under the 400-job limit. "
                               "Wait for slots; no existing jobs have been cancelled or changed.")
    receipt_root = ROOT / "results/rotnet_sweep_submissions"
    receipt_root.mkdir(parents=True, exist_ok=True)
    with directory_lock(receipt_root / ".submit.lock.d"):
        for old in receipt_root.glob("*/submission.json"):
            previous = json.loads(old.read_text())
            ids = list(previous.get("jobs", {}).values())
            if ids and not args.dry_run and not previous.get("dry_run"):
                active = {value.split("_", 1)[0] for value in queue}
                if active.intersection(ids):
                    raise RuntimeError(f"A sweep submission is still active: {old}")
        launch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = receipt_root / launch
        source = directory / "source"
        source.mkdir(parents=True)
        checksums = {}
        for name in SOURCE_FILES:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            checksums[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        for name in ("data", "checkpoints"):
            (ROOT / name).mkdir(exist_ok=True)
            (source / name).symlink_to(ROOT / name, target_is_directory=True)
        logs = ROOT / "slurm-logs"
        logs.mkdir(exist_ok=True)
        receipt = {"launch": launch, "source": str(source), "source_sha256": checksums,
                   "partition": args.partition, "gpu_type": args.gpu_type,
                   "excluded_nodes": EXCLUDE, "jobs": {}, "dry_run": args.dry_run}
        atomic_json(directory / "submission.json", receipt)
        common = ["sbatch", "--parsable", "--chdir=" + str(source),
                  "--output=" + str(logs / "%x-%A_%a.out"), "--exclude=" + EXCLUDE]
        gpu = ["--partition=" + args.partition, f"--gres=gpu:{args.gpu_type}:1"]
        environment = dict(os.environ, ECE661_PYTHON=args.python, ECE661_SOURCE=str(source))

        def submit(phase, options):
            command = [*common, "--job-name=p4-rotnet-" + phase, *options,
                       str(source / SWEEP_JOB), phase]
            print(" ".join(command), flush=True)
            if args.dry_run:
                job = {"preflight": "900001", "pilots": "900002", "train": "900003"}[phase]
            else:
                output = subprocess.run(command, check=True, capture_output=True,
                                        text=True, env=environment).stdout.strip()
                job = output.split(";", 1)[0]
                if not job.isdigit():
                    raise RuntimeError(f"Unexpected sbatch output: {output}")
            receipt["jobs"][phase] = job
            atomic_json(directory / "submission.json", receipt)
            return job

        preflight = submit("preflight", ["--partition=common", "--time=00:30:00"])
        pilot = submit("pilots", [*gpu, "--time=02:00:00", "--dependency=afterok:" + preflight])
        train = submit("train", [*gpu, "--time=2-00:00:00", f"--array=0-14%{args.concurrency}",
                                 "--dependency=afterok:" + pilot])
        print("Submission record:", directory / "submission.json")
        print(f"Preflight {preflight} -> pilot {pilot} -> 15-model array {train}")
        print("Only pretraining is submitted. Fine-tuning and linear evaluation are not queued.")


if __name__ == "__main__":
    main()
