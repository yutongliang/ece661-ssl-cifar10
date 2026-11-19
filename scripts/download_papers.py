#!/usr/bin/env python3
"""Download the plan's primary papers; no third-party packages required.

Run from any directory. Existing PDFs are preserved and verified by signature.
--check-only reports local availability without network access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from urllib.request import Request, urlopen


PAPER_DIR = Path(__file__).resolve().parents[1] / "papers"


def is_pdf(path):
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    papers = json.loads((PAPER_DIR / "manifest.json").read_text())
    failures = []
    for paper in papers:
        target = PAPER_DIR / paper["filename"]
        if not is_pdf(target) and not args.check_only:
            temporary = None
            try:
                request = Request(paper["url"], headers={"User-Agent": "ECE661-Paper-Downloader/1.0"})
                with urlopen(request, timeout=args.timeout) as response:
                    with tempfile.NamedTemporaryFile(dir=PAPER_DIR, suffix=".part", delete=False) as handle:
                        temporary = Path(handle.name)
                        while chunk := response.read(1024 * 1024):
                            handle.write(chunk)
                if not is_pdf(temporary):
                    raise ValueError("Server response is not a PDF")
                temporary.replace(target)
                print(f"Downloaded {paper['id']}: {target}")
            except Exception as error:
                print(f"FAILED {paper['id']}: {error}", file=sys.stderr)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if is_pdf(target):
            checksum = hashlib.sha256(target.read_bytes()).hexdigest()
            print(f"OK {paper['id']}: {target.name} ({target.stat().st_size:,} bytes; SHA256 {checksum})")
        else:
            failures.append(paper["id"])
            print(f"MISSING {paper['id']}: {target.name}", file=sys.stderr)
    print(f"Available: {len(papers) - len(failures)}/{len(papers)} PDFs")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
