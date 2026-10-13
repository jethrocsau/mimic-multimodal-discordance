"""Check one downloaded chunk against PhysioNet's SHA256SUMS.txt.

    python3 check_chunk.py <chunk list> <SHA256SUMS.txt> <download root> <ok out> <failed out>

A file is good when its SHA256 matches, or, if SHA256SUMS.txt has no entry for it, when it
is non-empty. Good paths are written (sorted) to <ok out>; the rest are appended to
<failed out>. Paths are relative to <download root>.
"""
import hashlib
import os
import sys

chunk_path, sums_path, root, ok_path, failed_path = sys.argv[1:]

with open(chunk_path) as f:
    chunk = f.read().split()
want = set(chunk)

expected = {}
with open(sums_path) as f:
    for line in f:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[1].strip() in want:
            expected[parts[1].strip()] = parts[0]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


ok, failed = [], []
for rel in chunk:
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        good = False
    elif rel in expected:
        good = sha256(path) == expected[rel]
    else:
        good = os.path.getsize(path) > 0
    (ok if good else failed).append(rel)

with open(ok_path, "w") as f:
    f.writelines(f"{rel}\n" for rel in sorted(ok))
with open(failed_path, "a") as f:
    f.writelines(f"{rel}\n" for rel in failed)
