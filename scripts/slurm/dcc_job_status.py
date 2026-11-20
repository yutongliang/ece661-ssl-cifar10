#!/usr/bin/env python3
"""Read-only summary of all projects in the current DCC user's Slurm account.

From the Mac, without uploading this script:
  ssh yl1064@dcc-login.oit.duke.edu 'python3 -' < scripts/slurm/dcc_job_status.py
"""
from collections import Counter, defaultdict
import getpass
import subprocess


def project(name):
    if name.startswith("p4-"):
        return "ECE661"
    if name.startswith(("cifar10_", "lrm_", "snnws", "mnist_")):
        return "ECE662"
    return "Other: " + name


def query(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Query failed: " + " ".join(command))
    return [[value.strip() for value in line.split("|")]
            for line in result.stdout.splitlines() if line.strip()]


def main():
    user = getpass.getuser()
    rows = query(["squeue", "--array", "-h", "-u", user,
                  "-o", "%i|%P|%j|%T|%M|%l|%R"])
    rows = [row for row in rows if len(row) == 7]
    totals = Counter((project(row[2]), row[3], row[1], row[6]) for row in rows)
    print("USER:", user)
    print("CURRENT QUEUE: expanded arrays; counts are tasks, not parent array IDs")
    print("PROJECT | STATE | TASKS | PARTITION | NODE / PENDING REASON")
    for (label, state, partition, reason), count in sorted(totals.items()):
        print(f"{label} | {state} | {count} | {partition} | {reason}")
    if not rows:
        print("No queued or running jobs.")
    print("Total queued/running tasks:", len(rows))

    details = defaultdict(list)
    for row in rows:
        if row[2].startswith("p4-") or row[2].startswith("lrm_"):
            key = (row[0].split("_", 1)[0], row[2], row[3], row[6])
            details[key].append(row)
    print("\nECE661 JOBS AND ECE662 CONTROLLERS")
    print("JOB/ARRAY | NAME | STATE | TASKS | ELAPSED / LIMIT | NODE / REASON")
    for (job, name, state, reason), members in sorted(details.items()):
        first = members[0]
        print(f"{job} | {name} | {state} | {len(members)} | "
              f"{first[4]} / {first[5]} | {reason}")

    print("\nLAST 24 HOURS: allocation records (including completed jobs)")
    try:
        recent = query(["sacct", "-X", "-n", "-P", "-u", user, "-S", "now-1day",
                        "--format=JobIDRaw,JobName%120,Partition,State,Elapsed,ExitCode"])
    except RuntimeError as error:
        print("Accounting query unavailable:", error)
        return
    recent = [row for row in recent if len(row) >= 6]
    counts = Counter((project(row[1]), row[3].split()[0]) for row in recent)
    print("PROJECT | STATE | RECORDS")
    for (label, state), count in sorted(counts.items()):
        print(f"{label} | {state} | {count}")
    failed = [row for row in recent if row[3].split()[0].rstrip("+") in
              {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "CANCELLED"}]
    print("\nUP TO 10 RECENT FAILURE/CANCELLATION RECORDS (may since have been retried)")
    for row in failed[-10:]:
        print(" | ".join(row[:6]))
    if not failed:
        print("None in the accounting response.")


if __name__ == "__main__":
    main()
