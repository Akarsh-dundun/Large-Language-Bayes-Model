"""Static schema checks for every FEMA NRI task JSON.

One set of checks runs for each task spec in
``tasks.build_fema_nri_tasks.TASK_SPECS``. Catches regressions in the task
JSON files without needing MCMC or a live LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tasks.build_fema_nri_tasks import TASK_SPECS, TASKS_DIR


EXPECTED_KEYS = {
  "name", "text", "data", "test_data", "targets",
  "true_latents", "task_type", "metadata",
}


@pytest.fixture(params=TASK_SPECS, ids=[s.name for s in TASK_SPECS])
def spec_and_task(request):
  spec = request.param
  path = TASKS_DIR / f"{spec.name}.json"
  with open(path) as f:
    task = json.load(f)
  return spec, task


def test_schema_fields_present(spec_and_task):
  spec, task = spec_and_task
  assert set(task.keys()) == EXPECTED_KEYS
  assert task["name"] == spec.name
  assert task["task_type"] == spec.task_type
  assert task["targets"] == list(spec.targets)
  assert task["true_latents"] is None
  assert isinstance(task["text"], str) and task["text"].strip()


def test_train_shape_and_dtype(spec_and_task):
  spec, task = spec_and_task
  values = task["data"][spec.value_name]
  assert len(values) == spec.n_train
  assert spec.n_train + spec.n_test <= 40
  expected_type = int if spec.value_dtype == "int" else float
  assert all(isinstance(v, expected_type) for v in values), (
    f"expected all {expected_type.__name__} in data[{spec.value_name}]"
  )
  if spec.positive_only:
    assert all(v > 0 for v in values)


def test_test_split_shape_and_disjointness(spec_and_task):
  spec, task = spec_and_task
  meta = task["metadata"]
  if spec.task_type == "prediction":
    assert task["test_data"] is not None
    test_values = task["test_data"][spec.value_name]
    assert len(test_values) == spec.n_test
    expected_type = int if spec.value_dtype == "int" else float
    assert all(isinstance(v, expected_type) for v in test_values)
    if spec.positive_only:
      assert all(v > 0 for v in test_values)
    train_fips = {c["stcofips"] for c in meta["counties_train"]}
    test_fips = {c["stcofips"] for c in meta["counties_test"]}
    assert len(train_fips) == spec.n_train
    assert len(test_fips) == spec.n_test
    assert train_fips.isdisjoint(test_fips)
  else:
    assert task["test_data"] is None
    assert meta["counties_test"] == []


def test_metadata_consistency(spec_and_task):
  spec, task = spec_and_task
  meta = task["metadata"]
  assert meta["n_train"] == spec.n_train
  assert meta["n_test"] == spec.n_test
  assert len(meta["counties_train"]) == spec.n_train
  assert len(meta["counties_test"]) == spec.n_test
  assert meta["attribute"] == spec.attribute
  assert meta["seed"] == spec.seed
  assert meta["value_dtype"] == spec.value_dtype
  if spec.state_filter:
    allowed = set(spec.state_filter)
    for entry in meta["counties_train"] + meta["counties_test"]:
      assert entry["state"] in allowed
