"""Shared paths, structured logs, configuration checks, and reproducibility metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGS = ROOT / "logs"
ANNOTATIONS = DATA / "annotations"


def load_config(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Expected a YAML object.")
    labels = [s["label"] for s in config["snapshots"]]
    if len(labels) != len(set(labels)) or len(labels) < 2:
        raise ValueError("Expected at least two snapshots with unique labels.")
    if config["experiment"]["top_k"] < 1:
        raise ValueError("top_k must be at least 1.")
    if config["preprocessing"]["overlap_words"] != 0:
        raise ValueError("Overlap not in the fixed design.")
    if config["embeddings"]["encoder"] != "sentence_transformer":
        raise ValueError("Scientific runs must use sentence_transformer encoder.")
    return config


def ensure_dirs() -> None:
    for path in (DATA, ANNOTATIONS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not existed:
            writer.writeheader()
        writer.writerows(rows)


def log_event(kind: str, **details) -> None:
    ensure_dirs()
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": kind, **details}
    with (LOGS / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(stable_json(row) + "\n")
    with (LOGS / "experiment.log").open("a", encoding="utf-8") as f:
        f.write(f"{row['timestamp_utc']} | {kind} | {stable_json(details)}\n")


def check_frozen_inputs(config_path: Path) -> dict:
    lock_path = ANNOTATIONS / "ground_truth.lock.json"
    if not lock_path.exists():
        raise RuntimeError("Missing ground_truth.lock.json. Complete annotations and run 'validate' first.")
    lock = read_json(lock_path)
    for label, file in [
        ("config_sha256", config_path),
        ("questions_sha256", ANNOTATIONS / "questions.csv"),
        ("ground_truth_sha256", ANNOTATIONS / "ground_truth.csv"),
        ("selection_sha256", DATA / "selected_articles.csv"),
        ("revision_manifest_sha256", DATA / "revision_manifest.csv"),
    ]:
        if not file.exists() or file_sha256(file) != lock.get(label):
            raise RuntimeError(f"Input '{label}' changed after freeze. Revalidate before continuing.")
    expected_labels = {s["label"] for s in load_config(config_path)["snapshots"]}
    if set(lock.get("processed_chunks_sha256", {})) != expected_labels:
        raise RuntimeError("Frozen lock does not cover every configured snapshot.")
    for snapshot_label, expected_hash in lock["processed_chunks_sha256"].items():
        path = DATA / "processed" / snapshot_label / "chunks.jsonl"
        if not path.exists() or file_sha256(path) != expected_hash:
            raise RuntimeError(f"Processed chunks for {snapshot_label} differ from the frozen ground truth.")
    return lock


def environment_report(config_path: Path) -> None:
    packages = ("numpy", "pandas", "PyYAML", "requests", "mwparserfromhell", "hnswlib", "sentence-transformers", "torch", "matplotlib", "pytest")
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    try:
        git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        git_revision = None
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "package_versions": versions,
        "config_sha256": file_sha256(config_path),
        "git_commit": git_revision,
        "configured_experiment_seed": load_config(config_path)["experiment"]["seed"],
        "machine": platform.machine(),
        "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted((ROOT / "temporal_hnsw").glob("*.py")) + [ROOT / "reproduce.py"]},
        "disclosure": "See AI_USE.md for the AI assistance disclosure.",
    }
    try:
        import torch
        report["torch_cuda_available"] = bool(torch.cuda.is_available())
        report["torch_cuda_device"] = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
    except ImportError:
        report["torch_cuda_available"] = None
    try:
        report["git_status_porcelain"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    except Exception:
        report["git_status_porcelain"] = None
    write_json(LOGS / "environment.json", report)
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (LOGS / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    except Exception as exc:
        log_event("pip_freeze_failed", error=str(exc))
    log_event("environment_recorded", config_sha256=report["config_sha256"], git_revision=git_revision)


