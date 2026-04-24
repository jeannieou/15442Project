"""
Generate a per-step chess trajectory report from stepsinfo.json.

Outputs:
1) A summary table for each step (move / speculation flags / predictions)
2) Detailed board state (after each move)
"""

import argparse
import json
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


def _build_records(step_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    board = chess.Board()
    records: List[Dict[str, Any]] = []

    for step_idx, step in _sorted_steps(step_info):
        player_id = step.get("player_id")
        player = "White" if player_id == 0 else "Black" if player_id == 1 else f"Player {player_id}"
        move_token = step.get("current_move")
        move_uci = _parse_move_token(move_token)
        prediction_moves = step.get("current_pred") or []
        speculation_moves = step.get("current_spec") or []

        prediction_hit = bool(move_token and move_token in prediction_moves)
        speculation_hit = bool(step.get("speculation_hit", False))

        legal = False
        error = None
        if move_uci is None:
            error = "missing_move"
        else:
            try:
                move = chess.Move.from_uci(move_uci)
                legal = move in board.legal_moves
                if legal:
                    board.push(move)
                else:
                    error = "illegal_move_for_reconstructed_board"
            except Exception as exc:
                error = f"move_parse_error: {exc}"

        records.append(
            {
                "step": step_idx,
                "player": player,
                "move": move_token,
                "move_uci": move_uci,
                "num_predictions": len(prediction_moves),
                "prediction_hit_this_step": prediction_hit,
                "speculation_hit_used_this_step": speculation_hit,
                "predictions": prediction_moves,
                "speculations": speculation_moves,
                "time_taken_current_agent": step.get("time_taken_current_agent"),
                "legal_on_reconstructed_board": legal,
                "reconstruct_error": error,
                "board_fen_after_move": board.fen(),
                "board_after_move": Utils.board_with_coords(board),
            }
        )

    return records


def _write_markdown(records: List[Dict[str, Any]], source_path: str, output_path: str) -> None:
    lines: List[str] = []
    lines.append("# Chess Trajectory Step Report")
    lines.append("")
    lines.append(f"- Source: `{source_path}`")
    lines.append(f"- Total steps: `{len(records)}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| step | player | move | spec_hit_used | pred_hit_this_step | #pred | legal |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for r in records:
        lines.append(
            f"| {r['step']} | {r['player']} | {r['move']} | "
            f"{int(r['speculation_hit_used_this_step'])} | {int(r['prediction_hit_this_step'])} | "
            f"{r['num_predictions']} | {int(r['legal_on_reconstructed_board'])} |"
        )

    lines.append("")
    lines.append("## Per-Step Details")
    lines.append("")

    for r in records:
        lines.append(f"### Step {r['step']}")
        lines.append(f"- player: `{r['player']}`")
        lines.append(f"- move: `{r['move']}`")
        lines.append(f"- speculate success used this step (`speculation_hit`): `{r['speculation_hit_used_this_step']}`")
        lines.append(f"- prediction hit this step (`current_move in current_pred`): `{r['prediction_hit_this_step']}`")
        lines.append(f"- predictions: `{r['predictions']}`")
        lines.append(f"- speculative responses: `{r['speculations']}`")
        lines.append(f"- legal on reconstructed board: `{r['legal_on_reconstructed_board']}`")
        if r["reconstruct_error"]:
            lines.append(f"- reconstruct error: `{r['reconstruct_error']}`")
        lines.append(f"- board FEN after move: `{r['board_fen_after_move']}`")
        lines.append("")
        lines.append("```text")
        lines.append(r["board_after_move"])
        lines.append("```")
        lines.append("")

    Utils.save_file("\n".join(lines), output_path, delete_prev_file=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a per-step trajectory report from stepsinfo.json.")
    parser.add_argument("--stepsinfo", "-i", default=None, help="Path to stepsinfo.json")
    parser.add_argument("--trajectory-dir", "-d", default=None, help="Path to trajectory directory containing stepsinfo.json")
    parser.add_argument("--output", "-o", default=None, help="Output markdown file path (default: <trajectory_dir>/step_report.md)")
    args = parser.parse_args()

    stepsinfo_path = _resolve_stepsinfo_path(args.stepsinfo, args.trajectory_dir)
    trajectory_dir = os.path.dirname(stepsinfo_path)
    output_path = args.output or os.path.join(trajectory_dir, "step_report.md")

    step_info = Utils.read_json(stepsinfo_path)
    records = _build_records(step_info)
    _write_markdown(records, stepsinfo_path, output_path)
    print(f"Saved step report to: {output_path}")


if __name__ == "__main__":
    main()
