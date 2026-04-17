"""Tests for generate_models.py and the seeding changes to llb.llm / llb.model_generator."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from generate_models import (
  GenerationConfig,
  GenerationResult,
  ModelCache,
  chunk_range,
  generate_and_cache,
  generate_chunk,
  merge_chunks,
)
from llb.llm import LLMClient
from llb.model_generator import (
  generate_models_with_diagnostics,
  MAX_ATTEMPTS_PER_MODEL,
)


def _make_config(**overrides):
  defaults = dict(
    text="coin flip bias",
    data={"flips": [0, 1, 1]},
    targets=["bias"],
    api_url="http://localhost:11434/api/generate",
    api_model="test-model:latest",
    n_models=6,
    random_seed=42,
  )
  defaults.update(overrides)
  return GenerationConfig(**defaults)


def _make_result(config=None, codes=None, diag=None):
  if config is None:
    config = _make_config()
  if codes is None:
    codes = [f"def model(data): pass  # model {i}" for i in range(config.n_models)]
  if diag is None:
    diag = {
      "requested_models": config.n_models,
      "generated_models": len(codes),
      "generation_failures": [],
      "invalid_generation_count": 0,
      "invalid_syntax_parsing_count": 0,
      "generation_request_failures": 0,
    }
  return GenerationResult(config=config, model_codes=codes, generation_diagnostics=diag)


# ─── GenerationConfig tests ──────────────────────────────────────

class TestGenerationConfig:
  def test_round_trip_dict(self):
    cfg = _make_config()
    restored = GenerationConfig.from_dict(cfg.to_dict())
    assert restored.text == cfg.text
    assert restored.data == cfg.data
    assert restored.targets == cfg.targets
    assert restored.api_model == cfg.api_model
    assert restored.n_models == cfg.n_models
    assert restored.random_seed == cfg.random_seed

  def test_round_trip_json(self, tmp_path):
    cfg = _make_config()
    path = tmp_path / "cfg.json"
    cfg.to_json(path)
    restored = GenerationConfig.from_json(path)
    assert restored.to_dict() == cfg.to_dict()

  def test_optional_fields_have_defaults(self):
    cfg = _make_config()
    assert cfg.api_key is None
    assert cfg.llm_timeout is None
    assert cfg.llm_max_retries == 2
    assert cfg.llm_retry_backoff == 2.0
    assert cfg.max_attempts_per_model == MAX_ATTEMPTS_PER_MODEL


# ─── GenerationResult tests ──────────────────────────────────────

class TestGenerationResult:
  def test_round_trip_dict(self):
    result = _make_result()
    d = result.to_dict()
    restored = GenerationResult.from_dict(d)
    assert restored.model_codes == result.model_codes
    assert restored.config.to_dict() == result.config.to_dict()
    assert restored.generation_diagnostics == result.generation_diagnostics

  def test_timestamp_populated(self):
    result = _make_result()
    assert result.timestamp is not None
    assert len(result.timestamp) > 0


# ─── ModelCache tests ─────────────────────────────────────────────

class TestModelCache:
  def test_save_load_round_trip(self, tmp_path):
    cache = ModelCache(tmp_path)
    result = _make_result()
    cache.save(result)
    assert cache.exists(result.config)
    loaded = cache.load(result.config)
    assert loaded.model_codes == result.model_codes
    assert loaded.config.to_dict() == result.config.to_dict()

  def test_exists_false_when_empty(self, tmp_path):
    cache = ModelCache(tmp_path)
    cfg = _make_config()
    assert not cache.exists(cfg)

  def test_list_cached(self, tmp_path):
    cache = ModelCache(tmp_path)
    r1 = _make_result(_make_config(n_models=3, random_seed=1))
    r2 = _make_result(_make_config(n_models=5, random_seed=2))
    cache.save(r1)
    cache.save(r2)
    cached = cache.list_cached()
    assert len(cached) == 2

  def test_cache_creates_directory(self, tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    cache = ModelCache(nested)
    assert nested.exists()


# ─── Cache key tests ──────────────────────────────────────────────

class TestCacheKey:
  def test_determinism(self, tmp_path):
    cache = ModelCache(tmp_path)
    cfg = _make_config()
    assert cache.cache_key(cfg) == cache.cache_key(cfg)

  def test_different_api_model(self, tmp_path):
    cache = ModelCache(tmp_path)
    k1 = cache.cache_key(_make_config(api_model="model-a"))
    k2 = cache.cache_key(_make_config(api_model="model-b"))
    assert k1 != k2

  def test_different_n_models(self, tmp_path):
    cache = ModelCache(tmp_path)
    k1 = cache.cache_key(_make_config(n_models=5))
    k2 = cache.cache_key(_make_config(n_models=10))
    assert k1 != k2

  def test_different_random_seed(self, tmp_path):
    cache = ModelCache(tmp_path)
    k1 = cache.cache_key(_make_config(random_seed=1))
    k2 = cache.cache_key(_make_config(random_seed=2))
    assert k1 != k2

  def test_different_text(self, tmp_path):
    cache = ModelCache(tmp_path)
    k1 = cache.cache_key(_make_config(text="problem A"))
    k2 = cache.cache_key(_make_config(text="problem B"))
    assert k1 != k2

  def test_different_data(self, tmp_path):
    cache = ModelCache(tmp_path)
    k1 = cache.cache_key(_make_config(data={"x": [1]}))
    k2 = cache.cache_key(_make_config(data={"x": [2]}))
    assert k1 != k2

  def test_different_targets(self, tmp_path):
    cache = ModelCache(tmp_path)
    k1 = cache.cache_key(_make_config(targets=["a"]))
    k2 = cache.cache_key(_make_config(targets=["b"]))
    assert k1 != k2

  def test_api_url_does_not_affect_key(self, tmp_path):
    cache = ModelCache(tmp_path)
    k1 = cache.cache_key(_make_config(api_url="http://host1:11434/api/generate"))
    k2 = cache.cache_key(_make_config(api_url="http://host2:11434/api/generate"))
    assert k1 == k2


# ─── Chunk range tests ────────────────────────────────────────────

class TestChunkRange:
  def test_even_split(self):
    ranges = [chunk_range(10, i, 5) for i in range(5)]
    starts = [r[0] for r in ranges]
    counts = [r[1] for r in ranges]
    assert sum(counts) == 10
    assert starts == sorted(starts)
    for i in range(len(ranges) - 1):
      assert starts[i] + counts[i] == starts[i + 1]

  def test_uneven_split(self):
    ranges = [chunk_range(10, i, 3) for i in range(3)]
    counts = [r[1] for r in ranges]
    assert sum(counts) == 10
    assert counts == [4, 3, 3]

  def test_single_task(self):
    start, count = chunk_range(7, 0, 1)
    assert start == 0
    assert count == 7

  def test_more_tasks_than_models(self):
    ranges = [chunk_range(3, i, 5) for i in range(5)]
    counts = [r[1] for r in ranges]
    assert sum(counts) == 3
    assert counts[3] == 0
    assert counts[4] == 0

  def test_no_gaps_or_overlaps(self):
    """Every model index 0..n_models-1 appears exactly once across all chunks."""
    n_models, n_tasks = 17, 5
    covered = set()
    for task_id in range(n_tasks):
      start, count = chunk_range(n_models, task_id, n_tasks)
      for idx in range(start, start + count):
        assert idx not in covered, f"Index {idx} appears in multiple chunks"
        covered.add(idx)
    assert covered == set(range(n_models))


# ─── Chunk save/load/merge tests ─────────────────────────────────

class TestChunkMerge:
  def test_save_load_chunks(self, tmp_path):
    cache = ModelCache(tmp_path)
    cfg = _make_config(n_models=9)
    for task_id in range(3):
      codes = [f"code_{task_id}_{i}" for i in range(3)]
      result = GenerationResult(
        config=cfg,
        model_codes=codes,
        generation_diagnostics={
          "requested_models": 3,
          "generated_models": 3,
          "generation_failures": [],
          "invalid_generation_count": 0,
          "invalid_syntax_parsing_count": 0,
          "generation_request_failures": 0,
        },
      )
      cache.save_chunk(result, task_id)

    chunks = cache.load_chunks(cfg)
    assert len(chunks) == 3

  def test_merge_concatenates_codes(self, tmp_path):
    cache = ModelCache(tmp_path)
    cfg = _make_config(n_models=6)
    all_expected_codes = []
    for task_id in range(3):
      codes = [f"code_{task_id}_{i}" for i in range(2)]
      all_expected_codes.extend(codes)
      result = GenerationResult(
        config=cfg,
        model_codes=codes,
        generation_diagnostics={
          "requested_models": 2,
          "generated_models": 2,
          "generation_failures": [],
          "invalid_generation_count": 0,
          "invalid_syntax_parsing_count": 0,
          "generation_request_failures": 0,
        },
      )
      cache.save_chunk(result, task_id)

    merged = cache.merge_chunks(cfg)
    assert merged.model_codes == all_expected_codes
    assert merged.generation_diagnostics["requested_models"] == 6
    assert merged.generation_diagnostics["generated_models"] == 6

  def test_merge_sums_diagnostics(self, tmp_path):
    cache = ModelCache(tmp_path)
    cfg = _make_config(n_models=6)
    for task_id in range(2):
      result = GenerationResult(
        config=cfg,
        model_codes=["code"],
        generation_diagnostics={
          "requested_models": 3,
          "generated_models": 1,
          "generation_failures": [(task_id, "some_error")],
          "invalid_generation_count": 2,
          "invalid_syntax_parsing_count": 1,
          "generation_request_failures": 1,
        },
      )
      cache.save_chunk(result, task_id)

    merged = cache.merge_chunks(cfg)
    assert merged.generation_diagnostics["requested_models"] == 6
    assert merged.generation_diagnostics["generated_models"] == 2
    assert merged.generation_diagnostics["invalid_generation_count"] == 4
    assert merged.generation_diagnostics["invalid_syntax_parsing_count"] == 2
    assert merged.generation_diagnostics["generation_request_failures"] == 2
    assert len(merged.generation_diagnostics["generation_failures"]) == 2

  def test_merge_is_idempotent(self, tmp_path):
    cache = ModelCache(tmp_path)
    cfg = _make_config(n_models=4)
    for task_id in range(2):
      codes = [f"code_{task_id}_{i}" for i in range(2)]
      result = GenerationResult(
        config=cfg,
        model_codes=codes,
        generation_diagnostics={
          "requested_models": 2,
          "generated_models": 2,
          "generation_failures": [],
          "invalid_generation_count": 0,
          "invalid_syntax_parsing_count": 0,
          "generation_request_failures": 0,
        },
      )
      cache.save_chunk(result, task_id)

    merged1 = cache.merge_chunks(cfg)
    merged2 = cache.merge_chunks(cfg)
    assert merged1.model_codes == merged2.model_codes
    assert merged1.generation_diagnostics == merged2.generation_diagnostics

  def test_merge_no_chunks_raises(self, tmp_path):
    cache = ModelCache(tmp_path)
    cfg = _make_config()
    with pytest.raises(FileNotFoundError):
      cache.merge_chunks(cfg)


# ─── Seed computation tests ──────────────────────────────────────

class TestSeeding:
  def test_llm_client_passes_seed_to_ollama_payload(self):
    client = LLMClient(api_url="http://localhost:11434/api/generate", model="test")
    payload = client._build_payload("hello", seed=123)
    assert payload["seed"] == 123
    assert payload["options"]["temperature"] == client.temperature
    assert payload["options"]["seed"] == 123

  def test_llm_client_temperature_default_is_nonzero(self):
    # Sampling diversity across seeds requires temperature > 0 for Ollama.
    client = LLMClient(api_url="http://localhost:11434/api/generate", model="test")
    assert client.temperature > 0

  def test_llm_client_explicit_temperature(self):
    client = LLMClient(api_url="http://localhost:11434/api/generate", model="test", temperature=0.5)
    payload = client._build_payload("hello", seed=1)
    assert payload["options"]["temperature"] == 0.5

  def test_llm_client_no_seed_omits_field(self):
    client = LLMClient(api_url="http://localhost:11434/api/generate", model="test")
    payload = client._build_payload("hello", seed=None)
    assert "seed" not in payload

  def test_llm_client_openai_chat_seed(self):
    client = LLMClient(api_url="http://api.example.com/v1/chat/completions", model="gpt-4")
    payload = client._build_payload("hello", seed=99)
    assert payload["seed"] == 99

  def test_seed_formula(self):
    """Verify the per-call seed formula matches the plan."""
    base_seed = 100
    max_attempts = 4

    recorded_seeds = []

    class FakeLLM:
      def generate(self, messages, seed=None):
        recorded_seeds.append(seed)
        return '```python\nimport numpyro\nimport numpyro.distributions as dist\ndef model(data):\n  numpyro.deterministic("bias", 0.5)\n```'

    llm = FakeLLM()
    generate_models_with_diagnostics(
      llm, text="test", data={}, targets=["bias"],
      n_models=3, base_seed=base_seed, slot_offset=0,
      max_attempts_per_model=max_attempts,
    )

    # Each slot succeeds on the first attempt, so we get 3 seeds
    assert len(recorded_seeds) == 3
    for slot_idx in range(3):
      expected = base_seed + slot_idx * max_attempts + 0
      assert recorded_seeds[slot_idx] == expected

  def test_seed_with_retries(self):
    """When a slot needs retries, each attempt gets a distinct seed."""
    base_seed = 50
    max_attempts = 4
    recorded_seeds = []
    call_count = [0]

    class RetryLLM:
      def generate(self, messages, seed=None):
        recorded_seeds.append(seed)
        call_count[0] += 1
        if call_count[0] <= 2:
          return "garbage with no model"
        return '```python\nimport numpyro\nimport numpyro.distributions as dist\ndef model(data):\n  numpyro.deterministic("bias", 0.5)\n```'

    llm = RetryLLM()
    codes, diag = generate_models_with_diagnostics(
      llm, text="test", data={}, targets=["bias"],
      n_models=1, base_seed=base_seed, slot_offset=0,
      max_attempts_per_model=max_attempts,
    )
    assert len(codes) == 1
    assert recorded_seeds[0] == base_seed + 0 * max_attempts + 0
    assert recorded_seeds[1] == base_seed + 0 * max_attempts + 1
    assert recorded_seeds[2] == base_seed + 0 * max_attempts + 2

  def test_no_seed_when_base_seed_is_none(self):
    """When base_seed is None, no seed is passed to the LLM."""
    recorded_seeds = []

    class FakeLLM:
      def generate(self, messages, seed=None):
        recorded_seeds.append(seed)
        return '```python\nimport numpyro\nimport numpyro.distributions as dist\ndef model(data):\n  numpyro.deterministic("bias", 0.5)\n```'

    llm = FakeLLM()
    generate_models_with_diagnostics(
      llm, text="test", data={}, targets=["bias"],
      n_models=2, base_seed=None, slot_offset=0,
    )
    assert all(s is None for s in recorded_seeds)

  def test_chunked_seeds_match_single_process(self):
    """Seeds from chunked generation match single-process generation."""
    base_seed = 200
    max_attempts = MAX_ATTEMPTS_PER_MODEL
    n_models = 10
    n_tasks = 3

    single_seeds = []

    class SeedRecorder:
      def __init__(self):
        self.seeds = []
      def generate(self, messages, seed=None):
        self.seeds.append(seed)
        return '```python\nimport numpyro\nimport numpyro.distributions as dist\ndef model(data):\n  numpyro.deterministic("bias", 0.5)\n```'

    recorder_single = SeedRecorder()
    generate_models_with_diagnostics(
      recorder_single, text="test", data={}, targets=["bias"],
      n_models=n_models, base_seed=base_seed, slot_offset=0,
      max_attempts_per_model=max_attempts,
    )
    single_seeds = recorder_single.seeds

    chunked_seeds = []
    for task_id in range(n_tasks):
      start, count = chunk_range(n_models, task_id, n_tasks)
      recorder = SeedRecorder()
      generate_models_with_diagnostics(
        recorder, text="test", data={}, targets=["bias"],
        n_models=count, base_seed=base_seed, slot_offset=start,
        max_attempts_per_model=max_attempts,
      )
      chunked_seeds.extend(recorder.seeds)

    assert chunked_seeds == single_seeds


# ─── generate_and_cache tests ────────────────────────────────────

class TestGenerateAndCache:
  @patch("generate_models._run_generation")
  def test_generates_on_cache_miss(self, mock_gen, tmp_path):
    cfg = _make_config()
    mock_gen.return_value = (["code_a", "code_b"], {
      "requested_models": 6,
      "generated_models": 2,
      "generation_failures": [],
      "invalid_generation_count": 4,
      "invalid_syntax_parsing_count": 0,
      "generation_request_failures": 0,
    })
    result = generate_and_cache(cfg, cache_dir=str(tmp_path))
    assert mock_gen.called
    assert result.model_codes == ["code_a", "code_b"]

  @patch("generate_models._run_generation")
  def test_returns_cached_on_hit(self, mock_gen, tmp_path):
    cfg = _make_config()
    cache = ModelCache(tmp_path)
    existing = _make_result(cfg, codes=["cached_code"])
    cache.save(existing)

    result = generate_and_cache(cfg, cache_dir=str(tmp_path))
    assert not mock_gen.called
    assert result.model_codes == ["cached_code"]

  @patch("generate_models._run_generation")
  def test_force_regenerate_ignores_cache(self, mock_gen, tmp_path):
    cfg = _make_config()
    cache = ModelCache(tmp_path)
    existing = _make_result(cfg, codes=["cached_code"])
    cache.save(existing)

    mock_gen.return_value = (["fresh_code"], {
      "requested_models": 6,
      "generated_models": 1,
      "generation_failures": [],
      "invalid_generation_count": 0,
      "invalid_syntax_parsing_count": 0,
      "generation_request_failures": 0,
    })
    result = generate_and_cache(cfg, cache_dir=str(tmp_path), force_regenerate=True)
    assert mock_gen.called
    assert result.model_codes == ["fresh_code"]


# ─── generate_chunk tests ────────────────────────────────────────

class TestGenerateChunk:
  @patch("generate_models._run_generation")
  def test_chunk_calls_with_correct_offset_and_count(self, mock_gen, tmp_path):
    cfg = _make_config(n_models=10)
    mock_gen.return_value = (["code"], {
      "requested_models": 4,
      "generated_models": 1,
      "generation_failures": [],
      "invalid_generation_count": 0,
      "invalid_syntax_parsing_count": 0,
      "generation_request_failures": 0,
    })

    generate_chunk(cfg, task_id=0, n_tasks=3, cache_dir=str(tmp_path))
    args, kwargs = mock_gen.call_args
    assert args[0] is cfg
    assert kwargs["slot_offset"] == 0
    assert kwargs["n_models_override"] == 4

  @patch("generate_models._run_generation")
  def test_chunk_saves_to_cache(self, mock_gen, tmp_path):
    cfg = _make_config(n_models=6)
    mock_gen.return_value = (["code_0"], {
      "requested_models": 2,
      "generated_models": 1,
      "generation_failures": [],
      "invalid_generation_count": 0,
      "invalid_syntax_parsing_count": 0,
      "generation_request_failures": 0,
    })

    generate_chunk(cfg, task_id=1, n_tasks=3, cache_dir=str(tmp_path))
    cache = ModelCache(tmp_path)
    chunks = cache.load_chunks(cfg)
    assert len(chunks) == 1
    assert chunks[0].model_codes == ["code_0"]


# ─── merge_chunks top-level function test ─────────────────────────

class TestMergeChunksTopLevel:
  def test_merge_via_function(self, tmp_path):
    cfg = _make_config(n_models=4)
    cache = ModelCache(tmp_path)
    for tid in range(2):
      r = GenerationResult(
        config=cfg,
        model_codes=[f"code_{tid}"],
        generation_diagnostics={
          "requested_models": 2,
          "generated_models": 1,
          "generation_failures": [],
          "invalid_generation_count": 0,
          "invalid_syntax_parsing_count": 0,
          "generation_request_failures": 0,
        },
      )
      cache.save_chunk(r, tid)

    merged = merge_chunks(cfg, cache_dir=str(tmp_path))
    assert merged.model_codes == ["code_0", "code_1"]
    assert cache.exists(cfg)
