#!/usr/bin/env python3
"""Run KuaiRand starter-kit experiments and emit machine-readable results.

The starter kit remains the source of truth for loading and evaluation. This
bridge lets the Node dashboard execute it without duplicating the metric code.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def find_data_dir(requested):
    candidates = []
    if requested:
        candidates.append(Path(requested))
    elif os.environ.get("KUAI_DATA_DIR"):
        candidates.append(Path(os.environ["KUAI_DATA_DIR"]))
    else:
        candidates.extend(
            [
                Path("KuaiRand-Pure/data"),
                Path("kuairand-starter-kit/KuaiRand-Pure/data"),
            ]
        )
    required = {
        "video_features_basic_pure.csv",
        "user_features_pure.csv",
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    }
    for candidate in candidates:
        if candidate.is_dir() and required.issubset({p.name for p in candidate.iterdir()}):
            return candidate.resolve()
    return None


def find_kit_dir(data_dir):
    requested = os.environ.get("KUAI_KIT_DIR")
    candidates = [Path(requested)] if requested else []
    if data_dir:
        candidates.extend([data_dir.parent.parent, data_dir.parent])
    candidates.extend([Path("kuairand-starter-kit"), Path(".")])
    for candidate in candidates:
        if candidate and (candidate / "baseline.py").is_file() and (candidate / "evaluate.py").is_file():
            return candidate.resolve()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    if not data_dir:
        print(
            json.dumps(
                {
                    "status": "missing_dataset",
                    "message": "KuaiRand-Pure data files were not found.",
                    "requiredPath": "KuaiRand-Pure/data",
                    "requiredFiles": [
                        "video_features_basic_pure.csv",
                        "user_features_pure.csv",
                        "log_standard_4_08_to_4_21_pure.csv",
                        "log_standard_4_22_to_5_08_pure.csv",
                    ],
                }
            )
        )
        return 0

    kit_dir = find_kit_dir(data_dir)
    if not kit_dir:
        print(json.dumps({"status": "missing_starter_kit", "message": "baseline.py and evaluate.py were not found."}))
        return 0

    results = []
    for model in ("random", "pop", "fm", "autoscale"):
        if model == "autoscale":
            command = [
                args.python,
                str(Path(__file__).with_name("kuairand_autoscale.py")),
                "--data-dir",
                str(data_dir),
                "--kit-dir",
                str(kit_dir),
                "--epochs",
                str(max(1, min(args.epochs, 3))),
            ]
        else:
            command = [
                args.python,
                str(kit_dir / "baseline.py"),
                "--data_dir",
                str(data_dir),
                "--model",
                model,
            ]
            if model == "fm":
                command.extend(["--epochs", str(args.epochs)])
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=kit_dir, capture_output=True, text=True, timeout=900)
        elapsed = round(time.perf_counter() - started, 3)
        output = completed.stdout + completed.stderr
        results.append({"model": model, "status": "ok" if completed.returncode == 0 else "error", "seconds": elapsed, "log": output[-12000:]})
        if completed.returncode != 0:
            break

    payload = {
        "status": "completed" if all(item["status"] == "ok" for item in results) else "error",
        "dataDir": str(data_dir),
        "kitDir": str(kit_dir),
        "models": results,
        "note": "These are real starter-kit runs when the supplied data directory contains the full KuaiRand-Pure dataset.",
    }
    report_path = Path(__file__).with_name("runs") / "benchmark_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), **payload}, indent=2) + "\n")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
