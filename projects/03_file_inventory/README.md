# File inventory

I wanted a simple way to notice content changes without trusting modification times. I walk a directory, hash files in bounded chunks and store relative paths in a versioned JSON inventory. Snapshot writes use a sibling temporary file followed by atomic replacement.

From the repository root, after the main setup:

```bash
python projects/03_file_inventory/main.py
pytest projects/03_file_inventory/tests -q
```

Example output:

```text
Modified: notes.txt
```

The default demonstration creates temporary data and detects a changed file. I test additions, removals, equal-length edits, empty files, malformed snapshots and symlink skipping. To save an inventory I use `--root /path/to/data --output /path/to/snapshot.json`; the output must be outside the scanned directory. The `compare` function compares two loaded snapshots. I skip links and special files. This is a local change detector, not a backup tool or a security boundary against concurrent filesystem changes.
