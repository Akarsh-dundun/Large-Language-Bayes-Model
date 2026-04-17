"""Model generation, caching, and distributed chunk support.

Provides infrastructure to generate probabilistic models from an LLM,
cache them to disk, and distribute generation across Slurm job arrays.
"""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llb.llm import LLMClient
from llb.model_generator import (
  generate_models_with_diagnostics,
  MAX_ATTEMPTS_PER_MODEL,
)


@dataclass
class GenerationConfig:
  text: str
  data: dict
  targets: list[str]
  api_url: str
  api_model: str
  n_models: int
  random_seed: int
  api_key: str | None = None
  llm_timeout: int | None = None
  llm_max_retries: int = 2
  llm_retry_backoff: float = 2.0
  max_attempts_per_model: int = MAX_ATTEMPTS_PER_MODEL

  def to_dict(self):
    return asdict(self)

  @classmethod
  def from_dict(cls, d):
    return cls(**d)

  def to_json(self, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
      json.dump(self.to_dict(), f, indent=2, sort_keys=True)

  @classmethod
  def from_json(cls, path):
    with open(path) as f:
      return cls.from_dict(json.load(f))


@dataclass
class GenerationResult:
  config: GenerationConfig
  model_codes: list[str]
  generation_diagnostics: dict[str, Any]
  timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

  def to_dict(self):
    return {
      "config": self.config.to_dict(),
      "model_codes": self.model_codes,
      "generation_diagnostics": _serialize_diagnostics(self.generation_diagnostics),
      "timestamp": self.timestamp,
    }

  @classmethod
  def from_dict(cls, d):
    return cls(
      config=GenerationConfig.from_dict(d["config"]),
      model_codes=d["model_codes"],
      generation_diagnostics=d["generation_diagnostics"],
      timestamp=d["timestamp"],
    )


def _serialize_diagnostics(diag):
  """Make diagnostics JSON-safe by converting tuples to lists."""
  out = {}
  for k, v in diag.items():
    if isinstance(v, list):
      out[k] = [list(item) if isinstance(item, tuple) else item for item in v]
    else:
      out[k] = v
  return out


class ModelCache:
  def __init__(self, cache_dir="model_cache"):
    self.cache_dir = Path(cache_dir)
    self.cache_dir.mkdir(parents=True, exist_ok=True)

  def cache_key(self, config):
    """Deterministic hash from the fields that affect generated output."""
    key_data = json.dumps({
      "text": config.text,
      "data": config.data,
      "targets": sorted(config.targets),
      "api_model": config.api_model,
      "n_models": config.n_models,
      "random_seed": config.random_seed,
    }, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]

  def _merged_path(self, config):
    return self.cache_dir / f"merged_{self.cache_key(config)}.json"

  def _chunk_path(self, config, task_id):
    return self.cache_dir / f"chunk_{self.cache_key(config)}_task{task_id}.json"

  def exists(self, config):
    return self._merged_path(config).exists()

  def save(self, result):
    path = self._merged_path(result.config)
    with open(path, "w") as f:
      json.dump(result.to_dict(), f, indent=2)
    return path

  def load(self, config):
    path = self._merged_path(config)
    with open(path) as f:
      return GenerationResult.from_dict(json.load(f))

  def save_chunk(self, result, task_id):
    path = self._chunk_path(result.config, task_id)
    with open(path, "w") as f:
      json.dump(result.to_dict(), f, indent=2)
    return path

  def load_chunks(self, config):
    key = self.cache_key(config)
    pattern = f"chunk_{key}_task*.json"
    chunks = []
    for path in sorted(self.cache_dir.glob(pattern)):
      with open(path) as f:
        chunks.append(GenerationResult.from_dict(json.load(f)))
    return chunks

  def merge_chunks(self, config):
    chunks = self.load_chunks(config)
    if not chunks:
      raise FileNotFoundError(
        f"No chunks found for cache key {self.cache_key(config)} "
        f"in {self.cache_dir}"
      )

    all_codes = []
    merged_diag = {
      "requested_models": 0,
      "generated_models": 0,
      "generation_failures": [],
      "invalid_generation_count": 0,
      "invalid_syntax_parsing_count": 0,
      "generation_request_failures": 0,
    }

    for chunk in chunks:
      all_codes.extend(chunk.model_codes)
      d = chunk.generation_diagnostics
      merged_diag["requested_models"] += d.get("requested_models", 0)
      merged_diag["generated_models"] += d.get("generated_models", 0)
      merged_diag["generation_failures"].extend(d.get("generation_failures", []))
      merged_diag["invalid_generation_count"] += d.get("invalid_generation_count", 0)
      merged_diag["invalid_syntax_parsing_count"] += d.get("invalid_syntax_parsing_count", 0)
      merged_diag["generation_request_failures"] += d.get("generation_request_failures", 0)

    merged = GenerationResult(
      config=config,
      model_codes=all_codes,
      generation_diagnostics=merged_diag,
      timestamp=datetime.now(timezone.utc).isoformat(),
    )
    self.save(merged)
    return merged

  def list_cached(self):
    results = []
    for path in sorted(self.cache_dir.glob("merged_*.json")):
      with open(path) as f:
        results.append(GenerationResult.from_dict(json.load(f)))
    return results


def chunk_range(n_models, task_id, n_tasks):
  """Compute (start, count) for a given task_id in an n_tasks split."""
  base_size = n_models // n_tasks
  remainder = n_models % n_tasks
  if task_id < remainder:
    start = task_id * (base_size + 1)
    count = base_size + 1
  else:
    start = remainder * (base_size + 1) + (task_id - remainder) * base_size
    count = base_size
  return start, count


def _run_generation(config, slot_offset=0, n_models_override=None):
  """Internal helper that creates an LLMClient and generates models."""
  llm_kwargs = {
    "api_url": config.api_url,
    "api_key": config.api_key,
    "model": config.api_model,
    "max_retries": config.llm_max_retries,
    "retry_backoff": config.llm_retry_backoff,
  }
  if config.llm_timeout is not None:
    llm_kwargs["timeout"] = config.llm_timeout
  llm = LLMClient(**llm_kwargs)

  n = n_models_override if n_models_override is not None else config.n_models
  codes, diag = generate_models_with_diagnostics(
    llm=llm,
    text=config.text,
    data=config.data,
    targets=config.targets,
    n_models=n,
    base_seed=config.random_seed,
    slot_offset=slot_offset,
    max_attempts_per_model=config.max_attempts_per_model,
  )
  return codes, diag


def generate_and_cache(config, cache_dir="model_cache", force_regenerate=False):
  cache = ModelCache(cache_dir)
  if not force_regenerate and cache.exists(config):
    return cache.load(config)

  codes, diag = _run_generation(config)
  result = GenerationResult(
    config=config,
    model_codes=codes,
    generation_diagnostics=diag,
  )
  cache.save(result)
  return result


def generate_chunk(config, task_id, n_tasks, cache_dir="model_cache"):
  start, count = chunk_range(config.n_models, task_id, n_tasks)
  if count <= 0:
    raise ValueError(
      f"Task {task_id} has no models to generate "
      f"(n_models={config.n_models}, n_tasks={n_tasks})"
    )

  codes, diag = _run_generation(config, slot_offset=start, n_models_override=count)
  result = GenerationResult(
    config=config,
    model_codes=codes,
    generation_diagnostics=diag,
  )
  cache = ModelCache(cache_dir)
  cache.save_chunk(result, task_id)
  return result


def merge_chunks(config, cache_dir="model_cache"):
  cache = ModelCache(cache_dir)
  return cache.merge_chunks(config)


def _cli():
  parser = argparse.ArgumentParser(
    description="Generate, cache, and merge LLM probabilistic models."
  )
  parser.add_argument(
    "--config", required=True,
    help="Path to GenerationConfig JSON file",
  )
  parser.add_argument(
    "--cache-dir", default="model_cache",
    help="Directory for cached model files (default: model_cache)",
  )

  subparsers = parser.add_subparsers(dest="command", required=True)

  sub_gen = subparsers.add_parser(
    "generate", help="Generate all models in a single process"
  )
  sub_gen.add_argument(
    "--force", action="store_true",
    help="Regenerate even if a cached result exists",
  )

  sub_chunk = subparsers.add_parser(
    "generate-chunk", help="Generate one chunk (for Slurm array tasks)"
  )
  sub_chunk.add_argument("--task-id", type=int, required=True)
  sub_chunk.add_argument("--n-tasks", type=int, required=True)

  subparsers.add_parser("merge", help="Merge all chunks into a single result")

  args = parser.parse_args()
  config = GenerationConfig.from_json(args.config)

  if args.command == "generate":
    result = generate_and_cache(config, cache_dir=args.cache_dir, force_regenerate=args.force)
    print(f"Generated {len(result.model_codes)} models")
    print(f"Saved to {ModelCache(args.cache_dir)._merged_path(config)}")

  elif args.command == "generate-chunk":
    result = generate_chunk(
      config, task_id=args.task_id, n_tasks=args.n_tasks,
      cache_dir=args.cache_dir,
    )
    print(f"Task {args.task_id}: generated {len(result.model_codes)} models")

  elif args.command == "merge":
    result = merge_chunks(config, cache_dir=args.cache_dir)
    print(f"Merged {len(result.model_codes)} models from chunks")
    print(f"Saved to {ModelCache(args.cache_dir)._merged_path(config)}")


if __name__ == "__main__":
  _cli()
