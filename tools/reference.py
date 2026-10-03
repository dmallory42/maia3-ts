"""Run the official Maia-3 5M model on fixed positions and write a parity fixture.

Output: test/fixtures/maia-reference.json

Inputs follow the official UCI engine in --use-uci-history mode: the history is
rebuilt from UCI moves played from the standard start position. Cases without
history pad with the current position, which is also what the engine does by
default (without --use-uci-history).
"""

import json
import random
import sys
from pathlib import Path

import chess
import torch

from maia3.dataset import get_legal_moves_mask
from maia3.uci import Maia3UCIEngine, parse_args

from download import checkpoint_path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "test" / "fixtures" / "maia-reference.json"
ELOS = [600, 1100, 1500, 2000, 2500]

# Opening and middlegame lines played from the start position (history used).
LINES = {
    "startpos": "",
    "e4": "e2e4",
    "ruy-lopez": "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7",
    "sicilian-najdorf": "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",
    "queens-gambit": "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8",
    "white-ep": "e2e4 g8f6 e4e5 d7d5",
    "black-ep": "g1f3 d7d5 g2g3 d5d4 e2e4",
    "both-ep-options": "e2e4 a7a6 e4e5 a6a5 d2d4 d7d5",
    "scholars-mate-threat": "e2e4 e7e5 f1c4 b8c6 d1h5",
    "fools-mate-in-one": "f2f3 e7e5 g2g4",
    "italian-long": "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d4 e5d4 c3d4 c5b4 b1c3 f6e4 e1g1 b4c3 d4d5 c3f6 f1e1 c6e7 e1e4 d7d6",
    "caro-kann": "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h6",
    "kings-indian": "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5",
    "black-can-castle-long": "d2d4 d7d5 c1f4 c8f5 b1c3 b8c6 d1d2 d8d7 e2e3 e7e6",
    "after-long-castle": "d2d4 d7d5 c1f4 c8f5 b1c3 b8c6 d1d2 d8d7 e2e3 e7e6 e1c1",
    "white-can-castle-both": "d2d4 d7d5 c1f4 c8f5 b1c3 b8c6 d1d2 d8d7 g1f3 g8f6 e2e3 e7e6",
}

# Positions given only as FEN (no history).
FENS = {
    "fen-start": chess.STARTING_FEN,
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "kiwipete-black": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "castle-k-only": "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w Kq - 4 8",
    "castle-black-only": "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R4RK1 b kq - 5 8",
    "promo-white": "8/P7/8/8/8/8/5k2/K7 w - - 0 1",
    "promo-white-capture": "1r5k/P7/8/8/8/8/8/K7 w - - 0 1",
    "promo-black": "7k/8/8/8/8/8/p7/4K3 b - - 0 1",
    "promo-black-capture": "k7/8/8/8/8/7K/6p1/5R2 b - - 0 1",
    "promo-many": "r3k3/1PP3P1/8/8/8/8/1p4p1/R3K2R w KQq - 0 1",
    "promo-many-black": "r3k3/1PP3P1/8/8/8/8/1p4p1/R3K2R b KQq - 0 1",
    "ep-white-fen": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "ep-black-fen": "rnbqkbnr/pp1ppppp/8/8/2pPP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 3",
    "kp-endgame": "8/8/4k3/8/4PK2/8/8/8 w - - 0 1",
    "kp-endgame-black": "8/8/4k3/8/4PK2/8/8/8 b - - 0 1",
    "rook-endgame": "8/5pk1/6p1/8/3R4/6P1/r4PK1/8 w - - 0 1",
    "lucena": "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
    "philidor": "4k3/8/8/3KP3/8/8/r7/7R b - - 0 1",
    "queen-vs-rook": "8/8/8/4k3/8/8/2Q5/K3r3 w - - 0 1",
    "bishop-knight-mate": "8/8/8/8/8/2k5/8/K1B1N3 w - - 0 1",
    "in-check-black": "rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 3",
    "knight-on-f7": "rnbqk2r/pppp1Npp/5n2/2b5/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 5",
    "knight-promo-fork": "3q3k/4P3/8/8/8/8/8/K7 w - - 0 1",
    "mate-in-one-white": "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "mate-in-one-black": "3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1",
    "tactics-middlegame": "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8",
    "open-middlegame-black": "r2q1rk1/pb1nbppp/1p2pn2/2pp4/2PP4/1PN1PN2/PB2BPPP/R2Q1RK1 b - - 3 10",
    "queen-endgame-black": "8/8/8/8/8/2k5/7q/K7 b - - 0 1",
    "stalemate-trap": "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
}

# Seeded random playouts from the start position, for varied middle and endgames.
RANDOM_PLAYOUTS = [(1, 20), (2, 35), (3, 50), (4, 70), (5, 90), (6, 110), (7, 30), (8, 60),
                   (9, 45), (10, 80), (11, 25), (12, 100), (13, 55), (14, 65), (15, 40)]


def random_line(seed, plies):
    rng = random.Random(seed)
    board = chess.Board()
    moves = []
    for _ in range(plies):
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        if not legal:
            break
        # Prefer captures and promotions a bit so playouts reach simpler positions.
        weighted = [m for m in legal if board.is_capture(m) or m.promotion] * 3 + legal
        move = rng.choice(weighted)
        board.push(move)
        moves.append(move.uci())
        if board.is_game_over():
            moves.pop()
            break
    return " ".join(moves)


def build_engine():
    argv = ["--model", "maia3-5m", "--checkpoint-path", str(checkpoint_path()),
            "--device", "cpu", "--no-use-amp", "--use-uci-history"]
    cfg = parse_args(argv)
    engine = Maia3UCIEngine(cfg)
    engine.ensure_model_loaded()
    return engine


@torch.no_grad()
def evaluate(engine, position_cmd, self_elo, oppo_elo):
    """Return (policy keyed by real UCI moves, WDL) for the position, via the engine's own helpers."""
    engine.cmd_position(position_cmd)
    tokens = engine._tokens_from_history(engine.history).unsqueeze(0)
    logits_move, logits_value, _ = engine.model(
        tokens, torch.tensor([self_elo]), torch.tensor([oppo_elo]))
    mask = get_legal_moves_mask(engine.board, engine.all_moves_dict)
    logits = logits_move[0].double().masked_fill(~mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    policy = {}
    for idx in torch.nonzero(mask).flatten().tolist():
        move = engine._move_from_index(idx)
        policy[move.uci()] = probs[idx].item()
    assert len(policy) == engine.board.legal_moves.count()
    loss, draw, win = torch.softmax(logits_value[0].double(), dim=-1).tolist()
    return policy, {"win": win, "draw": draw, "loss": loss}


def main():
    engine = build_engine()
    positions = []
    for name, line in LINES.items():
        positions.append((name, line.split() if line else [], None))
    for seed, plies in RANDOM_PLAYOUTS:
        line = random_line(seed, plies)
        positions.append((f"random-{seed}-{plies}", line.split(), None))
    for name, fen in FENS.items():
        positions.append((name, None, fen))

    cases = []
    for name, moves, fen in positions:
        if moves is not None:
            cmd = "position startpos" + (" moves " + " ".join(moves) if moves else "")
        else:
            cmd = f"position fen {fen}"
        for elo in ELOS:
            pairs = [(elo, elo)]
            if elo == 1500:
                pairs += [(1100, 2000), (2000, 1100)]
            for self_elo, oppo_elo in pairs:
                policy, value = evaluate(engine, cmd, self_elo, oppo_elo)
                board = engine.board
                assert board.is_valid(), name
                if not policy:
                    print(f"skip {name}: no legal moves", file=sys.stderr)
                    break
                case = {
                    "name": name,
                    "fen": board.fen(en_passant="fen"),
                    "eloSelf": self_elo,
                    "eloOppo": oppo_elo,
                    "policy": policy,
                    "value": value,
                }
                if moves is not None:
                    case["history"] = moves
                cases.append(case)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fixture = {
        "model": "UofTCSSLab/Maia3-5M maia3-5m.pt",
        "maia3Commit": "1e13597c42d4858b7cfd7cfdae01e297263364b2",
        "note": "history = UCI moves from the standard start position (engine --use-uci-history). "
                "Cases without history pad with the current position. value is from the side to move.",
        "cases": cases,
    }
    OUT.write_text(json.dumps(fixture, indent=None, separators=(",", ":")))
    n_pos = len({c["name"] for c in cases})
    print(f"wrote {len(cases)} cases over {n_pos} positions to {OUT}")


if __name__ == "__main__":
    main()
