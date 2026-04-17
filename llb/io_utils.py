"""Atomic I/O helpers shared by Stage A and Stage B.

All writes go through a tmp file plus os.replace so a preempted job never
leaves a partially written artifact behind.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def atomic_write_bytes(path, payload: bytes):
  """Write raw bytes to path via a tmp file and os.replace."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with open(tmp, "wb") as f:
    f.write(payload)
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, path)


def atomic_write_text(path, text: str):
  atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, payload):
  atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=False))


def atomic_savez(path, **arrays):
  """Atomically write a compressed npz."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with open(tmp, "wb") as f:
    np.savez_compressed(f, **arrays)
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, path)


def append_jsonl_atomic(path, record):
  """Append one JSON record to a jsonl file using a single write call.

  POSIX guarantees writes under PIPE_BUF (typically 4096 bytes) are atomic
  for files opened in append mode, which is enough for our short index rows.
  """
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  line = json.dumps(record) + "\n"
  with open(path, "a") as f:
    f.write(line)
    f.flush()


def sanitize_site_key(prefix: str, name: str) -> str:
  """Round-trip-safe key for np.savez (no slashes in dict keys)."""
  safe = name.replace("/", "__").replace("\\", "__")
  return f"{prefix}__{safe}"
