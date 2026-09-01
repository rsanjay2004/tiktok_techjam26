#!/usr/bin/env python3
"""CPU-first AutoScaleRec experiments for KuaiRand-Pure.

This script keeps the official evaluator untouched and trains small, auditable
ranking candidates from the standard log. The validation split selects the
winner; test is reported once for that winner.
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428), "test": (20220429, 20220508)}
BEHAVIORS = ("is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate")


def load_rows(data_dir):
    video_author = {}
    with open(Path(data_dir) / "video_features_basic_pure.csv", newline="") as stream:
        for row in csv.DictReader(stream):
            video_author[row["video_id"]] = row["author_id"]

    rows = []
    for filename in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(Path(data_dir) / filename, newline="") as stream:
            for row in csv.DictReader(stream):
                behaviors = [float(row.get(name, 0) or 0) for name in BEHAVIORS]
                duration = max(float(row["duration_ms"]), 1.0)
                satisfaction = (
                    float(row["long_view"])
                    + 0.5 * behaviors[1]
                    + 0.8 * behaviors[2]
                    + 0.5 * behaviors[3]
                    + 0.4 * behaviors[4]
                    - 0.8 * behaviors[5]
                )
                rows.append(
                    {
                        "date": int(row["date"]),
                        "time": int(row.get("time_ms") or row["date"]),
                        "user": row["user_id"],
                        "video": row["video_id"],
                        "author": video_author.get(row["video_id"], "UNK"),
                        "tab": row.get("tab", "UNK"),
                        "duration": duration,
                        "label": float(row["long_view"]),
                        "behaviors": behaviors,
                        "satisfaction": satisfaction,
                    }
                )
    return {
        name: [row for row in rows if low <= row["date"] <= high]
        for name, (low, high) in SPLITS.items()
    }


def smoothed(stats, index, global_rate, prior=20.0):
    count, values = stats
    return (values[index] + prior * global_rate[index]) / (count + prior)


def build_features(splits):
    train = sorted(splits["train"], key=lambda row: row["time"])
    global_count = max(len(train), 1)
    global_values = np.zeros(8, dtype=np.float64)
    for row in train:
        global_values[0] += row["label"]
        global_values[1:7] += row["behaviors"]
        global_values[7] += row["satisfaction"]
    global_values /= global_count

    def make_stats():
        return [0.0, np.zeros(8, dtype=np.float64)]

    item_stats = defaultdict(make_stats)
    author_stats = defaultdict(make_stats)
    user_tab_stats = defaultdict(make_stats)
    user_video_stats = defaultdict(make_stats)
    for row in train:
        for table, key in ((item_stats, row["video"]), (author_stats, row["author"]), (user_tab_stats, (row["user"], row["tab"])), (user_video_stats, (row["user"], row["video"]))):
            table[key][0] += 1
            table[key][1][0] += row["label"]
            table[key][1][1:7] += row["behaviors"]
            table[key][1][7] += row["satisfaction"]

    # Exponentially decayed histories. Querying at row time prevents future
    # interactions from leaking into a training or validation feature.
    user_author_decay = defaultdict(lambda: [0.0, 0.0, 0])
    user_video_decay = defaultdict(lambda: [0.0, 0.0, 0])
    tau = 3.0 * 24.0 * 60.0 * 60.0 * 1000.0
    train_history = {}

    def decayed_value(state, timestamp):
        if not state[2]:
            return 0.0
        factor = math.exp(-max(0, timestamp - state[2]) / tau)
        return (state[0] * factor) / max(state[1] * factor, 1.0)

    def update_decay(state, row):
        if state[2]:
            factor = math.exp(-max(0, row["time"] - state[2]) / tau)
            state[0] *= factor
            state[1] *= factor
        state[0] += row["satisfaction"]
        state[1] += 1.0
        state[2] = row["time"]

    for row in train:
        train_history[id(row)] = (
            decayed_value(user_author_decay[(row["user"], row["author"])], row["time"]),
            decayed_value(user_video_decay[(row["user"], row["video"])], row["time"]),
        )
        update_decay(user_author_decay[(row["user"], row["author"])], row)
        update_decay(user_video_decay[(row["user"], row["video"])], row)

    def feature_row(row, history_values=None):
        item = smoothed(item_stats[row["video"]], 0, global_values)
        author = smoothed(author_stats[row["author"]], 0, global_values)
        item_multi = smoothed(item_stats[row["video"]], 7, global_values)
        author_multi = smoothed(author_stats[row["author"]], 7, global_values)
        tab = smoothed(user_tab_stats[(row["user"], row["tab"])], 0, global_values)
        video_history = smoothed(user_video_stats[(row["user"], row["video"])], 0, global_values)
        if history_values is None:
            history_values = (
                decayed_value(user_author_decay.get((row["user"], row["author"]), (0.0, 0.0, 0)), row["time"]),
                decayed_value(user_video_decay.get((row["user"], row["video"]), (0.0, 0.0, 0)), row["time"]),
            )
        hour = ((int(row["time"]) // 3_600_000) % 24) / 24.0 if row["time"] else 0.5
        return [
            item,
            item_multi,
            author,
            author_multi,
            history_values[0],
            history_values[1],
            video_history,
            tab,
            min(row["duration"] / 300000.0, 1.0),
            math.sin(hour * 2 * math.pi),
            math.cos(hour * 2 * math.pi),
        ]

    encoded = {}
    for name, rows in splits.items():
        values = [feature_row(row, train_history[id(row)]) for row in rows] if name == "train" else [feature_row(row) for row in rows]
        encoded[name] = (
            np.asarray(values, dtype=np.float32),
            np.asarray([row["label"] for row in rows], dtype=np.float32),
            [row["user"] for row in rows],
            np.asarray([row["behaviors"] for row in rows], dtype=np.float32),
        )
    return encoded


def evaluate(kit_dir, users, labels, scores):
    sys.path.insert(0, str(kit_dir))
    from evaluate import evaluate as official_evaluate

    result = official_evaluate(users, labels, scores)
    return {key: float(value) if isinstance(value, (np.floating, float)) else int(value) for key, value in result.items()}


def train_fm_predictions(data_dir, kit_dir, epochs=11):
    """Train the starter-kit FM and expose scores for safe ensembling."""
    sys.path.insert(0, str(kit_dir))
    from baseline import FM
    from data import encode, load

    official_splits = load(data_dir)
    encoded, dimension = encode(official_splits)
    Xtr, ytr, _ = encoded["train"]
    Xva, _, _ = encoded["valid"]
    Xte, _, _ = encoded["test"]
    model = FM(dimension, k=16, lr=0.001, seed=0)
    rng = np.random.default_rng(0)
    valid_users = [row[1] for row in official_splits["valid"]]
    valid_labels = np.asarray([row[6] for row in official_splits["valid"]], dtype=np.float32)
    best_state = None
    best_score = -1.0
    for _ in range(epochs):
        order = rng.permutation(len(ytr))
        for start in range(0, len(order), 8192):
            index = order[start:start + 8192]
            model.step(Xtr[index], ytr[index])
        score = evaluate(kit_dir, valid_users, valid_labels, model.predict(Xva))["primary"]
        if score > best_score:
            best_score = score
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
    model.V, model.W, model.b = best_state
    return model.predict(Xva), model.predict(Xte), best_score


def pairwise_fit(X, y, users, seed=0, epochs=3, pairs_per_epoch=120000, kit_dir=None, valid_users=None, valid_labels=None, valid_features=None, verbose=False, l2=1e-4, hard_negative=True):
    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-6
    Z = (X - mean) / scale
    groups = defaultdict(list)
    for index, user in enumerate(users):
        groups[user].append(index)
    eligible = []
    for indices in groups.values():
        positives = [i for i in indices if y[i] > 0.5]
        negatives = [i for i in indices if y[i] <= 0.5]
        if positives and negatives:
            eligible.append((positives, negatives))
    rng = np.random.default_rng(seed)
    positive_indices = np.flatnonzero(y > 0.5)
    negative_indices = np.flatnonzero(y <= 0.5)
    weights = (Z[positive_indices].mean(axis=0) - Z[negative_indices].mean(axis=0)).astype(np.float32)
    for epoch in range(epochs):
        for _ in range(min(pairs_per_epoch, len(eligible) * 20)):
            positives, negatives = eligible[int(rng.integers(len(eligible)))]
            hard_negatives = rng.choice(negatives, size=min(5, len(negatives)), replace=False)
            positive = Z[rng.choice(positives)]
            negative_index = hard_negatives[np.argmax(Z[hard_negatives] @ weights)] if hard_negative else rng.choice(negatives)
            negative = Z[negative_index]
            delta = positive - negative
            margin = float(np.dot(weights, delta))
            gradient = 1.0 / (1.0 + math.exp(min(30.0, max(-30.0, margin))))
            weights += 0.01 * (gradient * delta - l2 * weights)
        if verbose and kit_dir is not None and valid_users is not None and valid_labels is not None and valid_features is not None:
            result = evaluate(kit_dir, valid_users, valid_labels, valid_features @ weights)
            print(f"pairwise epoch {epoch + 1}: valid GAUC {result['GAUC']:.4f} primary {result['primary']:.4f} item_feature={weights[0]:.6f}")
    if verbose:
        print(f"pairwise weights: norm={np.linalg.norm(weights):.6f} item_feature={weights[0]:.6f}")
    return mean, scale, weights


def ple_fit(X, targets, mean, scale, epochs=2, experts_count=3):
    """Train a small progressive-layered-style mixture of task experts."""
    Z = (X - mean) / scale
    # Keep this exploratory head bounded; the official FM still trains on all rows.
    limit = min(len(Z), 250000)
    Z, targets = Z[:limit], targets[:limit]
    rng = np.random.default_rng(0)
    experts = rng.normal(0, 0.01, (Z.shape[1], experts_count)).astype(np.float32)
    task_heads = np.zeros((experts_count, targets.shape[1]), dtype=np.float32)
    gate = np.zeros((Z.shape[1], experts_count), dtype=np.float32)
    for _ in range(epochs):
        for start in range(0, len(Z), 8192):
            batch = Z[start:start + 8192]
            truth = targets[start:start + 8192]
            latent = batch @ experts
            gate_logits = np.clip(batch @ gate, -12, 12)
            gate_prob = np.exp(gate_logits - gate_logits.max(axis=1, keepdims=True))
            gate_prob /= gate_prob.sum(axis=1, keepdims=True)
            logits = (latent * gate_prob) @ task_heads
            prediction = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            error = (prediction - truth) / max(len(batch), 1)
            mixed_experts = latent * gate_prob
            task_heads -= 0.08 * (mixed_experts.T @ error + 1e-4 * task_heads)
            latent_gradient = (error @ task_heads.T) * gate_prob
            experts -= 0.03 * (batch.T @ latent_gradient + 1e-4 * experts)
            gate_gradient = latent_gradient * latent
            gate -= 0.01 * (batch.T @ (gate_prob * (gate_gradient - (gate_gradient * gate_prob).sum(axis=1, keepdims=True))) + 1e-4 * gate)
    return experts, task_heads, gate


def ple_predict(Z, experts, task_heads, gate):
    latent = Z @ experts
    gate_logits = np.clip(Z @ gate, -12, 12)
    gate_prob = np.exp(gate_logits - gate_logits.max(axis=1, keepdims=True))
    gate_prob /= gate_prob.sum(axis=1, keepdims=True)
    return (latent * gate_prob) @ task_heads


def auxiliary_fit(X, targets, mean, scale, epochs=1):
    """Train shared linear auxiliary heads for click/like/follow/etc."""
    Z = (X - mean) / scale
    heads = np.zeros((Z.shape[1], targets.shape[1]), dtype=np.float32)
    for _ in range(epochs):
        for start in range(0, len(Z), 8192):
            batch = Z[start:start + 8192]
            truth = targets[start:start + 8192]
            logits = np.clip(batch @ heads, -20, 20)
            prediction = 1.0 / (1.0 + np.exp(-logits))
            heads -= 0.03 * ((batch.T @ (prediction - truth)) / max(len(batch), 1) + 1e-4 * heads)
    return heads


def tune_feature_blend(Z, users, labels, kit_dir, seed=0, trials=12):
    """Select a feature blend on validation only; test is never used here."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(trials):
        weights = rng.normal(0, 1, Z.shape[1]).astype(np.float32)
        result = evaluate(kit_dir, users, labels, Z @ weights)
        if best is None or result["primary"] > best["result"]["primary"]:
            best = {"weights": weights, "result": result}
    return best


def run(args):
    started = time.perf_counter()
    splits = load_rows(args.data_dir)
    features = build_features(splits)
    Xtr, ytr, utr, atr = features["train"]
    Xva, yva, uva, ava = features["valid"]
    Xte, yte, ute, ate = features["test"]
    diagnostic_mean = Xtr.mean(axis=0)
    diagnostic_scale = Xtr.std(axis=0) + 1e-6
    diagnostic_Zva = (Xva - diagnostic_mean) / diagnostic_scale
    if args.verbose:
        raw_item = evaluate(args.kit_dir, uva, yva, diagnostic_Zva[:, 0])
        print(f"raw item ablation: valid GAUC {raw_item['GAUC']:.4f} nDCG@5 {raw_item['nDCG@5']:.4f} primary {raw_item['primary']:.4f}")
    mean, scale, weights = pairwise_fit(
        Xtr, ytr, utr, epochs=args.epochs, kit_dir=args.kit_dir,
        valid_users=uva, valid_labels=yva, valid_features=diagnostic_Zva, verbose=args.verbose,
        l2=args.pairwise_l2, hard_negative=not args.uniform_negatives,
    )
    auxiliary_heads = auxiliary_fit(Xtr, atr, mean, scale)
    ple_experts, ple_heads, ple_gate = ple_fit(Xtr, atr, mean, scale)
    Ztr, Zva, Zte = (Xtr - mean) / scale, (Xva - mean) / scale, (Xte - mean) / scale
    auxiliary_mix = np.asarray([0.15, 0.15, 0.2, 0.15, 0.15, -0.2], dtype=np.float32)
    auxiliary_valid = Zva @ auxiliary_heads @ auxiliary_mix
    auxiliary_test = Zte @ auxiliary_heads @ auxiliary_mix
    ple_valid = ple_predict(Zva, ple_experts, ple_heads, ple_gate) @ auxiliary_mix
    ple_test = ple_predict(Zte, ple_experts, ple_heads, ple_gate) @ auxiliary_mix

    candidates = {
        "multi_behavior": 0.25 * Zva[:, 0] + 0.75 * Zva[:, 1] + 0.25 * Zva[:, 2] + 0.75 * Zva[:, 3] + auxiliary_valid,
        "time_decay": 0.25 * Zva[:, 0] + 0.25 * Zva[:, 2] + 1.0 * Zva[:, 4] + 0.9 * Zva[:, 5],
        "pairwise_ranking": Zva @ weights,
        "auxiliary_multitask": auxiliary_valid,
        "ple_multitask": ple_valid,
        "retrieval_then_rank": 0.35 * (0.6 * Zva[:, 0] + 0.4 * Zva[:, 4]) + 0.4 * (Zva @ weights) + 0.25 * auxiliary_valid,
    }
    tuned = tune_feature_blend(Zva, uva, yva, args.kit_dir)
    candidates["validation_tuned_blend"] = Zva @ tuned["weights"]
    fm_valid_raw, fm_test_raw, fm_valid_primary = train_fm_predictions(args.data_dir, args.kit_dir, epochs=args.fm_epochs)
    fm_mean = fm_valid_raw.mean()
    fm_scale = fm_valid_raw.std() + 1e-6
    fm_valid = (fm_valid_raw - fm_mean) / fm_scale
    fm_test = (fm_test_raw - fm_mean) / fm_scale
    custom_valid = candidates["retrieval_then_rank"]
    custom_test = 0.35 * (0.6 * Zte[:, 0] + 0.4 * Zte[:, 4]) + 0.4 * (Zte @ weights) + 0.25 * auxiliary_test
    custom_mean = custom_valid.mean()
    custom_scale = custom_valid.std() + 1e-6
    candidates["fm_baseline"] = fm_valid
    candidates["fm_plus_autoscale"] = 0.75 * fm_valid + 0.25 * ((custom_valid - custom_mean) / custom_scale)
    ple_mean = ple_valid.mean()
    ple_scale = ple_valid.std() + 1e-6
    candidates["fm_plus_ple"] = 0.75 * fm_valid + 0.25 * ((ple_valid - ple_mean) / ple_scale)
    valid_results = {name: evaluate(args.kit_dir, uva, yva, score) for name, score in candidates.items()}
    raw_winner = max(valid_results, key=lambda name: valid_results[name]["primary"])
    fm_primary = valid_results["fm_baseline"]["primary"]
    promotion_margin = 0.002
    winner = raw_winner if valid_results[raw_winner]["primary"] >= fm_primary + promotion_margin else "fm_baseline"
    test_scores = {
        "multi_behavior": 0.25 * Zte[:, 0] + 0.75 * Zte[:, 1] + 0.25 * Zte[:, 2] + 0.75 * Zte[:, 3] + auxiliary_test,
        "time_decay": 0.25 * Zte[:, 0] + 0.25 * Zte[:, 2] + 1.0 * Zte[:, 4] + 0.9 * Zte[:, 5],
        "pairwise_ranking": Zte @ weights,
        "auxiliary_multitask": auxiliary_test,
        "ple_multitask": ple_test,
        "retrieval_then_rank": 0.35 * (0.6 * Zte[:, 0] + 0.4 * Zte[:, 4]) + 0.4 * (Zte @ weights) + 0.25 * auxiliary_test,
        "validation_tuned_blend": Zte @ tuned["weights"],
        "fm_baseline": fm_test,
        "fm_plus_autoscale": 0.75 * fm_test + 0.25 * ((custom_test - custom_mean) / custom_scale),
        "fm_plus_ple": 0.75 * fm_test + 0.25 * ((ple_test - ple_mean) / ple_scale),
    }
    test_result = evaluate(args.kit_dir, ute, yte, test_scores[winner])
    return {
        "status": "completed",
        "winner": winner,
        "valid": valid_results,
        "test": test_result,
        "rows": {name: len(value[0]) for name, value in features.items()},
        "seconds": round(time.perf_counter() - started, 3),
        "selection": "winner selected by validation primary; test evaluated once after selection",
        "official_fm_validation_primary": fm_valid_primary,
        "raw_winner": raw_winner,
        "promotion_margin": promotion_margin,
        "promotion_rule": "A challenger must beat FM validation primary by at least 0.002 to become the final model.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--kit-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fm-epochs", type=int, default=11)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pairwise-l2", type=float, default=1e-4)
    parser.add_argument("--uniform-negatives", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args)))
