"""Atomic directory locks for shared cluster storage, without flock/NFS locks."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import uuid


def read_owner(path):
    # A contender can see mkdir before the owner has published its metadata.
    for _ in range(20):
        try:
            return json.loads((path / "owner.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.05)
    return None


def stale(owner):
    if not owner:
        return False
    if owner.get("host") == socket.gethostname():
        try:
            os.kill(int(owner["pid"]), 0)
        except ProcessLookupError:
            return True
        except (PermissionError, ValueError, KeyError):
            return False
        else:
            return False
    job = owner.get("slurm_job_id", "")
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job):
        return False
    if job == os.environ.get("SLURM_JOB_ID"):
        return int(os.environ.get("SLURM_RESTART_COUNT", "0")) > int(owner.get("restart", 0))
    # On a different node, a missing PID is not evidence of a dead owner.
    # Ask Slurm; a query failure is never treated as permission to steal a lock.
    try:
        result = subprocess.run(["squeue", "-h", "-j", job, "-o", "%i"],
                                capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return (not result.stdout.strip()
            and (result.returncode == 0 or "Invalid job id" in result.stderr))


def acquire(path, owner):
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            path.mkdir()
        except FileExistsError:
            previous = read_owner(path)
            if not stale(previous):
                raise RuntimeError(f"Lock is busy: {path}; owner={previous}. "
                                   "No jobs were submitted by this lock attempt.")
            # Serialize stale-owner recovery, then recheck after obtaining the gate.
            recovery = path.with_name(path.name + ".recovery")
            try:
                recovery.mkdir()
            except FileExistsError as error:
                raise RuntimeError(f"Another process is recovering {path}; retry shortly") from error
            try:
                if path.exists():
                    current = read_owner(path)
                    if not stale(current):
                        raise RuntimeError(f"Lock now has a live/unknown owner: {current}")
                    retired = path.with_name(path.name + ".stale-" + uuid.uuid4().hex)
                    path.rename(retired)
                    shutil.rmtree(retired)
            finally:
                recovery.rmdir()
            continue
        temporary = path / "owner.tmp"
        try:
            temporary.write_text(json.dumps(owner) + "\n")
            temporary.replace(path / "owner.json")
        except BaseException:
            shutil.rmtree(path)
            raise
        return
    raise RuntimeError(f"Could not acquire {path}; another process won the lock")


@contextmanager
def directory_lock(path):
    path = Path(path)
    owner = {"host": socket.gethostname(), "pid": os.getpid(), "token": uuid.uuid4().hex,
             "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
             "restart": int(os.environ.get("SLURM_RESTART_COUNT", "0"))}
    acquire(path, owner)
    try:
        yield
    finally:
        current = read_owner(path)
        if current and current.get("token") == owner["token"]:
            shutil.rmtree(path)
