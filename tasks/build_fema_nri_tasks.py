"""Build all FEMA NRI task JSONs used by the epistemic-uncertainty paper.

Pulls per-county attributes from the FEMA National Risk Index FeatureServer,
samples counties with a fixed seed per task, and writes one task JSON per
spec into this directory.

Run:
  uv run python -m tasks.build_fema_nri_tasks
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests


FEATURE_SERVER_URL = (
  "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/"
  "National_Risk_Index_Counties/FeatureServer/0/query"
)

TASKS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class FemaTaskSpec:
  name: str
  attribute: str
  state_filter: tuple[str, ...] | None
  positive_only: bool
  scale: float
  value_name: str
  value_dtype: str
  n_train: int
  n_test: int
  task_type: str
  targets: tuple[str, ...]
  text: str
  seed: int
  units: str
  extra_notes: str = ""


TASK_SPECS: list[FemaTaskSpec] = [
  FemaTaskSpec(
    name="hurricane_eal_counties",
    attribute="HRCN_EALT",
    state_filter=(
      "FL", "TX", "LA", "NC", "SC", "GA", "AL", "MS",
      "VA", "MD", "DE", "NJ", "NY", "CT", "MA", "ME",
    ),
    positive_only=True,
    scale=1e6,
    value_name="losses_musd",
    value_dtype="float",
    n_train=20,
    n_test=10,
    task_type="prediction",
    targets=("median_loss",),
    text=(
      "I have the expected annual hurricane-related economic losses, in "
      "millions of US dollars, for 20 hurricane-exposed US counties. "
      "Losses are strictly positive and vary across several orders of "
      "magnitude. Estimate the underlying loss distribution and predict "
      "losses for 10 held-out counties."
    ),
    seed=0,
    units="USD millions",
    extra_notes=(
      "IID assumption is approximate. Counties differ in population, "
      "building stock, and coastal exposure."
    ),
  ),
  FemaTaskSpec(
    name="tornado_counts_plains",
    attribute="TRND_EVNTS",
    state_filter=("KS", "OK", "NE", "TX"),
    positive_only=True,
    scale=1.0,
    value_name="events",
    value_dtype="int",
    n_train=20,
    n_test=0,
    task_type="estimation",
    targets=("rate",),
    text=(
      "I have the total number of recorded tornadoes in each of 20 "
      "counties in the US Great Plains. Each value is a nonnegative "
      "integer count accumulated over roughly three decades. Estimate "
      "the underlying tornado event rate per county."
    ),
    seed=1,
    units="count of recorded tornadoes",
    extra_notes=(
      "Counts are integer. Counties vary substantially in area, which "
      "induces overdispersion relative to a pooled Poisson."
    ),
  ),
  FemaTaskSpec(
    name="earthquake_frequency_west",
    attribute="ERQK_AFREQ",
    state_filter=("CA", "NV", "OR", "WA", "UT", "ID", "AK"),
    positive_only=True,
    scale=1.0,
    value_name="frequency_per_year",
    value_dtype="float",
    n_train=20,
    n_test=0,
    task_type="estimation",
    targets=("mean_frequency",),
    text=(
      "I have the annualized earthquake frequency, in events per year, "
      "for 20 counties in the western United States. Values are small "
      "positive numbers. Estimate the mean annual earthquake frequency."
    ),
    seed=2,
    units="events per year",
    extra_notes="Values are continuous positive rates derived from historical seismicity.",
  ),
  FemaTaskSpec(
    name="wildfire_eal_west",
    attribute="WFIR_EALT",
    state_filter=(
      "CA", "OR", "WA", "ID", "MT", "NV", "AZ", "NM", "UT", "CO", "WY",
    ),
    positive_only=True,
    scale=1e6,
    value_name="losses_musd",
    value_dtype="float",
    n_train=20,
    n_test=10,
    task_type="prediction",
    targets=("median_loss",),
    text=(
      "I have the expected annual wildfire-related economic losses, in "
      "millions of US dollars, for 20 counties in the western United "
      "States. Losses are strictly positive and span several orders of "
      "magnitude. Estimate the underlying loss distribution and predict "
      "losses for 10 held-out counties."
    ),
    seed=3,
    units="USD millions",
    extra_notes=(
      "IID assumption is approximate. Counties differ in vegetation, "
      "building stock, and historical fire activity."
    ),
  ),
  FemaTaskSpec(
    name="inland_flood_eal",
    attribute="IFLD_EALT",
    state_filter=None,
    positive_only=True,
    scale=1e6,
    value_name="losses_musd",
    value_dtype="float",
    n_train=20,
    n_test=10,
    task_type="prediction",
    targets=("median_loss",),
    text=(
      "I have the expected annual inland-flooding economic losses, in "
      "millions of US dollars, for 20 US counties drawn from across "
      "the country. Losses are strictly positive and span many orders "
      "of magnitude. Estimate the underlying loss distribution and "
      "predict losses for 10 held-out counties."
    ),
    seed=4,
    units="USD millions",
    extra_notes=(
      "Sampled nationwide without a state filter. Heavy-tailed dollar "
      "losses typical of hydrological hazards."
    ),
  ),
]


def _fetch_counties(attribute: str, state_filter, positive_only: bool):
  """Query the FeatureServer for one attribute and return attribute dicts."""
  clauses = []
  if state_filter:
    state_list = ",".join(f"'{s}'" for s in state_filter)
    clauses.append(f"STATEABBRV IN ({state_list})")
  if positive_only:
    clauses.append(f"{attribute} > 0")
  where = " AND ".join(clauses) if clauses else "1=1"

  params = {
    "where": where,
    "outFields": f"STCOFIPS,STATE,STATEABBRV,COUNTY,{attribute}",
    "f": "json",
    "returnGeometry": "false",
    "resultRecordCount": 5000,
  }
  resp = requests.get(FEATURE_SERVER_URL, params=params, timeout=60)
  resp.raise_for_status()
  payload = resp.json()
  features = payload.get("features", [])
  if not features:
    raise RuntimeError(
      f"No features returned for {attribute} with filter {state_filter}. "
      f"Payload keys: {list(payload.keys())}"
    )
  return [f["attributes"] for f in features]


def _coerce_value(raw, scale: float, value_dtype: str):
  if raw is None:
    return None
  scaled = raw / scale
  if value_dtype == "int":
    return int(round(scaled))
  return float(scaled)


def _county_meta(record):
  return {
    "state": record.get("STATEABBRV"),
    "county": record.get("COUNTY"),
    "stcofips": record.get("STCOFIPS"),
  }


def build_task(spec: FemaTaskSpec) -> dict:
  records = _fetch_counties(spec.attribute, spec.state_filter, spec.positive_only)
  records = [
    r for r in records
    if r.get(spec.attribute) is not None
    and (not spec.positive_only or r[spec.attribute] > 0)
  ]
  records.sort(key=lambda r: r["STCOFIPS"])

  n_total = spec.n_train + spec.n_test
  if len(records) < n_total:
    raise RuntimeError(
      f"Only {len(records)} records for {spec.name}; need {n_total}."
    )

  rng = np.random.default_rng(spec.seed)
  idx = rng.choice(len(records), size=n_total, replace=False)
  chosen = [records[i] for i in idx]

  values = [
    _coerce_value(r[spec.attribute], spec.scale, spec.value_dtype)
    for r in chosen
  ]

  train_values = values[: spec.n_train]
  test_values = values[spec.n_train : spec.n_train + spec.n_test]
  train_counties = [_county_meta(r) for r in chosen[: spec.n_train]]
  test_counties = [_county_meta(r) for r in chosen[spec.n_train : spec.n_train + spec.n_test]]

  data = {spec.value_name: train_values}
  if spec.n_test > 0:
    test_data = {spec.value_name: test_values}
  else:
    test_data = None

  metadata = {
    "internet_seen": True,
    "source": "FEMA National Risk Index v1.20 (Dec 2025)",
    "attribute": spec.attribute,
    "source_url": FEATURE_SERVER_URL,
    "units": spec.units,
    "scale_divisor": spec.scale,
    "value_dtype": spec.value_dtype,
    "seed": spec.seed,
    "state_filter": list(spec.state_filter) if spec.state_filter else None,
    "positive_only": spec.positive_only,
    "n_train": spec.n_train,
    "n_test": spec.n_test,
    "counties_train": train_counties,
    "counties_test": test_counties,
    "drawn_at": dt.date.today().isoformat(),
    "notes": spec.extra_notes,
  }

  return {
    "name": spec.name,
    "text": spec.text,
    "data": data,
    "test_data": test_data,
    "targets": list(spec.targets),
    "true_latents": None,
    "task_type": spec.task_type,
    "metadata": metadata,
  }


def write_task(spec: FemaTaskSpec) -> Path:
  task = build_task(spec)
  out_path = TASKS_DIR / f"{spec.name}.json"
  with open(out_path, "w") as f:
    json.dump(task, f, indent=2)

  train = task["data"][spec.value_name]
  test = (task["test_data"] or {}).get(spec.value_name, [])
  print(f"[{spec.name}] wrote {out_path.name}")
  print(
    f"  train n={len(train)} "
    f"min={min(train):.4g} max={max(train):.4g}"
  )
  if test:
    print(
      f"  test  n={len(test)} "
      f"min={min(test):.4g} max={max(test):.4g}"
    )
  return out_path


def main():
  for spec in TASK_SPECS:
    write_task(spec)


if __name__ == "__main__":
  main()
