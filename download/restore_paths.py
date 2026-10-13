"""Move files copied flat from Cloud Storage back to their relative paths.

    python3 restore_paths.py <chunk list> <flat dir> <download root>

`gcloud storage cp -I` puts every object in one folder under its base name; this moves
<flat dir>/<base name> to <download root>/<relative path> for each path in the chunk.
download_all.sh only uses Cloud Storage when base names are unique within a project.
"""
import os
import sys

chunk_path, flat, root = sys.argv[1:]

with open(chunk_path) as f:
    for rel in f.read().split():
        src = os.path.join(flat, os.path.basename(rel))
        if os.path.exists(src):
            dst = os.path.join(root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)
