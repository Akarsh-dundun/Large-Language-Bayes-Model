"""Per-shard run manifest written to ``<cell_dir>/_manifest/shard_<uuid>.json``.

Records git commit, library versions, host, UTC times, CLI args, and file
hashes so a reviewer can answer "exactly what produced this artifact?".
"""
from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import os
import platform
import socket
import subprocess
import sys
import uuid
from pathlib import Path

from .codes import manifest_dir
from .io_utils import atomic_write_json


def _safe_run(cmd):
  try:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    return out.stdout.strip() if out.returncode == 0 else None
  except Exception:
    return None


def _git_commit():
  return _safe_run(["git", "rev-parse", "HEAD"])


def _git_status_short():
  return _safe_run(["git", "status", "--short"])


def _import_version(module_name):
  try:
    import importlib
    mod = importlib.import_module(module_name)
    return getattr(mod, "__version__", None)
  except Exception:
    return None


def _file_sha256(path):
  try:
    h = hashlib.sha256()
    with open(path, "rb") as f:
      for chunk in iter(lambda: f.read(8192), b""):
        h.update(chunk)
    return h.hexdigest()
  except Exception:
    return None


def write_run_manifest(cell_dir, stage: str, cli_args: dict, extras: dict | None = None):
  """Write one ``shard_<uuid>.json`` per array task.

  Filename uses a fresh UUID so re-runs after preemption never overwrite an
  existing manifest. Append-only history of what each shard did.
  """
  cell_dir = Path(cell_dir)
  mdir = manifest_dir(cell_dir)
  mdir.mkdir(parents=True, exist_ok=True)

  shard_uid = uuid.uuid4().hex
  payload = {
    "stage": stage,
    "shard_uuid": shard_uid,
    "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "user": getpass.getuser(),
    "pid": os.getpid(),
    "python_version": sys.version,
    "platform": platform.platform(),
    "git_commit": _git_commit(),
    "git_status_short": _git_status_short(),
    "library_versions": {
      "jax": _import_version("jax"),
      "jaxlib": _import_version("jaxlib"),
      "numpyro": _import_version("numpyro"),
      "numpy": _import_version("numpy"),
      "scipy": _import_version("scipy"),
      "equinox": _import_version("equinox"),
    },
    "slurm": {
      "job_id": os.environ.get("SLURM_JOB_ID"),
      "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
      "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
      "node_list": os.environ.get("SLURM_JOB_NODELIST"),
    },
    "cli_args": _stringify(cli_args),
    "extras": _stringify(extras or {}),
  }

  task_path = (extras or {}).get("task_path")
  if task_path and task_path != "None":
    payload["task_file_sha256"] = _file_sha256(task_path)
  llm_path = (extras or {}).get("llm_config_path")
  if llm_path and llm_path != "None":
    payload["llm_config_file_sha256"] = _file_sha256(llm_path)

  out = mdir / f"shard_{shard_uid}.json"
  atomic_write_json(out, payload)
  return out


def _stringify(d):
  """Make sure the manifest is always JSON-serializable."""
  out = {}
  for k, v in (d or {}).items():
    if isinstance(v, (str, int, float, bool)) or v is None:
      out[k] = v
    elif isinstance(v, (list, tuple)):
      out[k] = [str(x) if not isinstance(x, (str, int, float, bool)) and x is not None else x for x in v]
    elif isinstance(v, dict):
      out[k] = _stringify(v)
    else:
      out[k] = str(v)
  return out
