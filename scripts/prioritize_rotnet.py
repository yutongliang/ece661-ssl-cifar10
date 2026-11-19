#!/usr/bin/env python3
"""Deprioritize only this user's pending ECE662 campaign jobs; never cancel jobs."""
import argparse
import getpass
import json
from pathlib import Path
import re
import subprocess
import sys

from rotnet_sweep import atomic_json


def pending_jobs():
    result = subprocess.run(["squeue", "--array", "-h", "-u", getpass.getuser(),
                             "-t", "PD", "-o", "%i|%j"],
                            check=True, capture_output=True, text=True)
    return [line.split("|", 1) for line in result.stdout.splitlines() if "|" in line]


def job_details(job):
    result = subprocess.run(["scontrol", "show", "job", "-o", job],
                            check=True, capture_output=True, text=True)
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)", result.stdout))


def eligible(job, name, details):
    return (job.isdigit() and name.startswith("cifar10_lrms3")
            and details.get("JobState") == "PENDING"
            and details.get("UserId", "").split("(")[0] == getpass.getuser()
            and details.get("JobName") == name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="otherwise only show candidates")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--restore", action="store_true", help="restore previously recorded Nice values")
    parser.add_argument("--nice", type=int, default=100,
                        help="modest target Nice value for a new explicit adjustment (default: 100)")
    args = parser.parse_args()
    if args.nice < 0:
        parser.error("nice must be nonnegative")
    if args.restore:
        entries = json.loads(args.record.read_text())
        for entry in entries:
            try:
                details = job_details(entry["job_id"])
                if not eligible(entry["job_id"], entry["name"], details):
                    continue
                if int(details.get("Nice", "0")) != entry["new_nice"]:
                    continue  # Respect any subsequent manual change.
                command = ["scontrol", "update", f"JobId={entry['job_id']}",
                           f"Nice={entry['old_nice']}"]
                print(" ".join(command))
                if args.apply:
                    subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                print(f"Could not restore {entry['job_id']}: {error}", file=sys.stderr)
        return
    if args.record.exists():
        raise ValueError("Priority record already exists; restore it or use a new record path")
    entries = []
    for job, name in pending_jobs():
        if not job.isdigit() or not name.startswith("cifar10_lrms3"):
            continue
        try:
            details = job_details(job)
            if not eligible(job, name, details):
                continue
            old = int(details.get("Nice", "0"))
            # Set the explicit target, including reducing an earlier excessive
            # positive Nice value. Do not retain a legacy Nice=1000000 setting.
            new = args.nice
            if old == new:
                continue
            entry = {"job_id": job, "name": name, "old_nice": old, "new_nice": new}
            if args.apply:
                # Record intent before the RPC so a process interruption is recoverable.
                entries.append(entry)
                atomic_json(args.record, entries)
                subprocess.run(["scontrol", "update", f"JobId={job}", f"Nice={new}"], check=True)
                after = job_details(job)
                entry["confirmed"] = int(after.get("Nice", "0")) == new
                atomic_json(args.record, entries)
                if not entry["confirmed"]:
                    print(f"Priority adjustment was not confirmed for {job}", file=sys.stderr)
            else:
                entries.append(entry)
        except subprocess.CalledProcessError as error:
            print(f"Could not adjust {job}: {error}", file=sys.stderr)
    count = sum(bool(entry.get("confirmed")) for entry in entries) if args.apply else len(entries)
    print(f"{'Confirmed adjusted' if args.apply else 'Would adjust'} {count} pending ECE662 jobs.")
    print("Running jobs and the lrm_seeds supervisor are not stopped or cancelled.")
    if args.apply and not args.record.exists():
        atomic_json(args.record, [])


if __name__ == "__main__":
    main()
