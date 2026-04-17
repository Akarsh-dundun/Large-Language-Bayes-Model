"""Code-file layout for Stage A and Stage B of the paper sweep.

Stage A writes one ``code_<sha>.code.json`` per distinct valid LLM-generated
NumPyro program. Hashes are first 16 hex characters of the sha256 of the
AST-canonicalized code so cosmetic differences (whitespace, comments) do not
spawn duplicate models. Stage B reads these files, runs inference, and writes
``sample_<sha>.{npz,meta.json}`` next door.
"""
from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .io_utils import atomic_write_text, append_jsonl_atomic


CODES_SUBDIR = "codes"
SAMPLES_SUBDIR = "samples"
MANIFEST_SUBDIR = "_manifest"
FAILURES_SUBDIR = "_failures"
CODE_INDEX_FILENAME = "_index.jsonl"


def canonicalize_code(raw_code: str) -> str:
  """Return ast.unparse(ast.parse(raw_code)).

  Collapses cosmetic differences (whitespace, comments). Preserves variable
  names, constants, and import order. Raises SyntaxError on un-parseable code.
  """
  return ast.unparse(ast.parse(raw_code))


def code_hash(canonical_code: str) -> str:
  """First 16 hex chars of sha256(canonical_code). 64 bits is plenty
  of collision resistance for 10,000 distinct codes per cell.
  """
  digest = hashlib.sha256(canonical_code.encode("utf-8")).hexdigest()
  return digest[:16]


def codes_dir(cell_dir) -> Path:
  return Path(cell_dir) / CODES_SUBDIR


def samples_dir(cell_dir) -> Path:
  return Path(cell_dir) / SAMPLES_SUBDIR


def manifest_dir(cell_dir) -> Path:
  return Path(cell_dir) / MANIFEST_SUBDIR


def failures_dir(cell_dir) -> Path:
  return codes_dir(cell_dir) / FAILURES_SUBDIR


def code_path(cell_dir, sha: str) -> Path:
  return codes_dir(cell_dir) / f"code_{sha}.code.json"


def sample_npz_path(cell_dir, sha: str) -> Path:
  return samples_dir(cell_dir) / f"sample_{sha}.npz"


def sample_meta_path(cell_dir, sha: str) -> Path:
  return samples_dir(cell_dir) / f"sample_{sha}.meta.json"


def code_index_path(cell_dir) -> Path:
  return codes_dir(cell_dir) / CODE_INDEX_FILENAME


def count_distinct_codes(cell_dir) -> int:
  """Number of distinct ``code_*.code.json`` files in the cell."""
  return len(list(codes_dir(cell_dir).glob("code_*.code.json")))


def list_code_hashes(cell_dir) -> list[str]:
  """Sorted list of sha hashes present under ``codes/``."""
  paths = sorted(codes_dir(cell_dir).glob("code_*.code.json"))
  return [p.stem.replace("code_", "").replace(".code", "") for p in paths]


def claim_code(
  cell_dir,
  sha: str,
  payload: dict,
) -> bool:
  """Atomically claim ``code_<sha>.code.json`` via O_EXCL.

  Returns True when the code was written (first time we see this hash) and
  False when the file already exists (duplicate of an earlier generation).
  """
  path = code_path(cell_dir, sha)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  text = json.dumps(payload, indent=2)

  # Use O_EXCL on the final path to atomically claim the hash. Two shards
  # racing on the same hash will see exactly one winner.
  fd = None
  try:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
  except FileExistsError:
    return False

  try:
    with os.fdopen(fd, "w") as f:
      f.write(text)
      f.flush()
      os.fsync(f.fileno())
    fd = None
    return True
  except Exception:
    # Best-effort cleanup so a write error does not leave a 0-byte file
    # blocking future claims.
    try:
      if fd is not None:
        os.close(fd)
    finally:
      try:
        os.unlink(str(path))
      except FileNotFoundError:
        pass
    raise


def append_code_index(cell_dir, record: dict):
  append_jsonl_atomic(code_index_path(cell_dir), record)


def write_failure(cell_dir, slot: int, record: dict):
  fdir = failures_dir(cell_dir)
  path = fdir / f"failure_{slot:08d}.json"
  atomic_write_text(path, json.dumps(record, indent=2))


def load_code_payload(cell_dir, sha: str) -> dict:
  with open(code_path(cell_dir, sha)) as f:
    return json.load(f)


@dataclass(frozen=True)
class CodePayload:
  """Convenience accessor for a Stage A code file."""
  sha: str
  raw_code: str
  canonical_code: str
  raw_llm_response: str
  prompt_messages: list
  first_seed: int
  generation_diagnostics: dict
  generation_seconds: float

  @classmethod
  def from_dict(cls, d: dict) -> "CodePayload":
    return cls(
      sha=d["sha"],
      raw_code=d["raw_code"],
      canonical_code=d["canonical_code"],
      raw_llm_response=d.get("raw_llm_response", ""),
      prompt_messages=d.get("prompt_messages", []),
      first_seed=int(d.get("first_seed", 0)),
      generation_diagnostics=d.get("generation_diagnostics", {}),
      generation_seconds=float(d.get("generation_seconds", 0.0)),
    )
