"""
Estimate speculative-hit probability from board-state heuristics.

This script replays a trajectory from stepsinfo.json and computes per-step
features + heuristic probability:
  - in_check
  - number of legal moves
  - fraction of checking moves
  - fraction of captures
  - promotions available
  - mate-in-1 available

Label used for offline evaluation (when available):
  prediction_hit_this_step = current_move in current_pred
Only defined on steps where current_pred is non-empty.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple

import chess

from utils import Utils


def _resolve_stepsinfo_path(stepsinfo: str | None, trajectory_dir: str | None) -> str:
    if stepsinfo:
        path = stepsinfo
    elif trajectory_dir:
        path = os.path.join(trajectory_dir, "stepsinfo.json")
    else:
        raise ValueError("Provide --stepsinfo or --trajectory-dir.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"stepsinfo.json not found: {path}")
    return path


def _parse_move_token(move_token: str | None) -> str | None:
    if not move_token:
        return None
    token = move_token.strip()
    if token.startswith("[") and token.endswith("]"):
        token = token[1:-1].strip()
    return token or None


def _sorted_steps(step_info: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    return sorted(((int(k), v) for k, v in step_info.items()), key=lambda x: x[0])


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _feature_counts(board: chess.Board) -> Dict[str, Any]:
    legal = list(board.legal_moves)
    n_legal = len(legal)

    checking_moves = 0
    capture_moves = 0
    promotion_moves = 0
    mate_in_1_moves = 0

    for m in legal:
        if board.is_capture(m):
            capture_moves += 1
        if m.promotion is not None:
            promotion_moves += 1
        b = board.copy()
        b.push(m)
        if b.is_check():
            checking_moves += 1
        if b.is_checkmate():
            mate_in_1_moves += 1

    return {
        "in_check": board.is_check(),
        "n_legal": n_legal,
        "checking_moves": checking_moves,
        "capture_moves": capture_moves,
        "promotion_moves": promotion_moves,
        "mate_in_1_moves": mate_in_1_moves,
        "checking_ratio": (checking_moves / n_legal) if n_legal else 0.0,
        "capture_ratio": (capture_moves / n_legal) if n_legal else 0.0,
    }


def _heuristic_delta(features: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    n_legal = features["n_legal"]
    in_check = features["in_check"]
    checking_ratio = features["checking_ratio"]
    capture_ratio = features["capture_ratio"]
    promotion_moves = features["promotion_moves"]
    mate_in_1_moves = features["mate_in_1_moves"]

    parts: Dict[str, float] = {}

    parts["in_check"] = 0.9 if in_check else 0.0

    if n_legal <= 3:
        parts["low_legal_moves"] = 1.1
    elif n_legal <= 8:
        parts["low_legal_moves"] = 0.6
    elif n_legal <= 15:
        parts["low_legal_moves"] = 0.2
    elif n_legal >= 35:
        parts["low_legal_moves"] = -0.6
    elif n_legal >= 25:
        parts["low_legal_moves"] = -0.3
    else:
        parts["low_legal_moves"] = 0.0

    parts["mate_in_1"] = 2.0 if mate_in_1_moves > 0 else 0.0
    parts["promotion"] = 0.8 if promotion_moves > 0 else 0.0
    parts["checking_ratio"] = 1.1 * min(checking_ratio, 0.5)
    parts["capture_ratio"] = 0.7 * min(capture_ratio, 0.7)

    delta = sum(parts.values())
    return delta, parts


def build_rows(step_info: Dict[str, Any], base_rate: float, threshold: float) -> List[Dict[str, Any]]:
    board = chess.Board()
    rows: List[Dict[str, Any]] = []
    base_logit = _logit(base_rate)

    for step_idx, step in _sorted_steps(step_info):
        move_token = step.get("current_move")
        move_uci = _parse_move_token(move_token)
        preds = step.get("current_pred") or []

        label = None
        if len(preds) > 0:
            label = 1 if move_token in preds else 0

        features = _feature_counts(board)
        delta, parts = _heuristic_delta(features)
        prob = _sigmoid(base_logit + delta)
        should_speculate = prob >= threshold

        rows.append(
            {
                "step": step_idx,
                "player_id": step.get("player_id"),
                "move": move_token,
                "num_predictions": len(preds),
                "prediction_hit_this_step": label,
                "speculation_hit_used_this_step": bool(step.get("speculation_hit", False)),
                "heuristic_probability": prob,
                "heuristic_should_speculate": should_speculate,
                "features": features,
                "score_parts": parts,
            }
        )

        if move_uci:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
            else:
                break

    return rows


def _fit_base_rate_from_labeled(step_info: Dict[str, Any]) -> float:
    labeled: List[int] = []
    for _, step in _sorted_steps(step_info):
        preds = step.get("current_pred") or []
        if len(preds) > 0:
            labeled.append(1 if step.get("current_move") in preds else 0)
    if not labeled:
        return 0.25
    return sum(labeled) / len(labeled)


def summarize(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    labeled = [r for r in rows if r["prediction_hit_this_step"] is not None]
    if not labeled:
        return {"labeled_windows": 0}

    y = [int(r["prediction_hit_this_step"]) for r in labeled]
    p = [float(r["heuristic_probability"]) for r in labeled]
    d = [bool(r["heuristic_should_speculate"]) for r in labeled]

    tp = sum(1 for yi, di in zip(y, d) if yi == 1 and di)
    fp = sum(1 for yi, di in zip(y, d) if yi == 0 and di)
    tn = sum(1 for yi, di in zip(y, d) if yi == 0 and not di)
    fn = sum(1 for yi, di in zip(y, d) if yi == 1 and not di)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(y) if y else 0.0

    return {
        "labeled_windows": len(y),
        "empirical_hit_rate": sum(y) / len(y),
        "mean_heuristic_probability": sum(p) / len(p),
        "threshold": threshold,
        "decision_rate": sum(1 for x in d if x) / len(d),
        "precision_if_decide": precision,
        "recall_if_decide": recall,
        "decision_accuracy": accuracy,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic speculative-hit probability from stepsinfo.json")
    parser.add_argument("--stepsinfo", "-i", default=None, help="Path to stepsinfo.json")
    parser.add_argument("--trajectory-dir", "-d", default=None, help="Trajectory directory containing stepsinfo.json")
    parser.add_argument(
        "--base-rate",
        type=float,
        default=None,
        help="Base hit-rate prior in [0,1]. If omitted, fit from labeled windows in this trajectory.",
    )
    parser.add_argument("--threshold", type=float, default=0.35, help="Decision threshold for should_speculate")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    args = parser.parse_args()

    stepsinfo_path = _resolve_stepsinfo_path(args.stepsinfo, args.trajectory_dir)
    trajectory_dir = os.path.dirname(stepsinfo_path)
    output_path = args.output or os.path.join(trajectory_dir, "heuristic_hit_report.json")

    step_info = Utils.read_json(stepsinfo_path)
    base_rate = args.base_rate if args.base_rate is not None else _fit_base_rate_from_labeled(step_info)
    base_rate = min(max(base_rate, 1e-4), 1 - 1e-4)

    rows = build_rows(step_info, base_rate=base_rate, threshold=args.threshold)
    summary = summarize(rows, threshold=args.threshold)

    report = {
        "source": stepsinfo_path,
        "base_rate": base_rate,
        "summary": summary,
        "rows": rows,
    }
    Utils.save_json(report, output_path, delete_prev_file=True)
    print(f"Saved heuristic report to: {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
