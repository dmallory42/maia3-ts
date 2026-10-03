// Board and move encoding for the Maia-3 port. Maia-3 is by CSSLab,
// https://github.com/CSSLab/maia3, licensed AGPL-3.0.
//
// Squares are indexed a1 = 0 ... h8 = 63 (rank * 8 + file). The model always sees the
// board from the side to move: when black is to move, ranks are flipped and colours
// swapped. Each history position is encoded from its own side to move.

const START_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";
const PIECE_CHANNEL: Record<string, number> = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };

/** Piece placement (64 chars, "." for empty, a1 first) plus side to move. */
export interface Position {
  squares: string[];
  whiteToMove: boolean;
}

export function squareIndex(name: string): number {
  return (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97);
}

function parsePlacement(placement: string): string[] {
  const squares = new Array<string>(64).fill(".");
  const ranks = placement.split("/");
  if (ranks.length !== 8) throw new Error(`maia: bad FEN placement "${placement}"`);
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of ranks[r]!) {
      if (ch >= "1" && ch <= "8") file += Number(ch);
      else squares[(7 - r) * 8 + file++] = ch;
    }
  }
  return squares;
}

export function positionFromFen(fen: string): Position {
  const [placement, turn] = fen.trim().split(/\s+/);
  return { squares: parsePlacement(placement ?? ""), whiteToMove: turn !== "b" };
}

/** Applies a UCI move without legality checks (castling, en passant and promotion aware). */
function applyUci(pos: Position, uci: string): Position {
  const sq = pos.squares.slice();
  const from = squareIndex(uci.slice(0, 2));
  const to = squareIndex(uci.slice(2, 4));
  const piece = sq[from]!;
  if (piece === ".") throw new Error(`maia: history move ${uci} has no piece on its from square`);
  const lower = piece.toLowerCase();
  const fileDelta = (to & 7) - (from & 7);
  if (lower === "k" && Math.abs(fileDelta) === 2) {
    // Castling: move the rook too.
    const rank = from & 56;
    const [rookFrom, rookTo] = fileDelta > 0 ? [rank + 7, rank + 5] : [rank, rank + 3];
    sq[rookTo] = sq[rookFrom]!;
    sq[rookFrom] = ".";
  } else if (lower === "p" && fileDelta !== 0 && sq[to] === ".") {
    // En passant: remove the pawn beside the from square.
    sq[(from & 56) + (to & 7)] = ".";
  }
  sq[from] = ".";
  sq[to] = uci.length > 4 ? (pos.whiteToMove ? uci[4]!.toUpperCase() : uci[4]!.toLowerCase()) : piece;
  return { squares: sq, whiteToMove: !pos.whiteToMove };
}

/**
 * Returns the last `count` positions (oldest first, current last). `history` is the list of
 * UCI moves from the standard start position; replaying it must reach `current`.
 */
export function historyPositions(current: Position, history: string[] | undefined, count: number): Position[] {
  if (!history) return [current];
  let pos: Position = { squares: parsePlacement(START_PLACEMENT), whiteToMove: true };
  const out: Position[] = [pos];
  for (const uci of history) {
    pos = applyUci(pos, uci);
    out.push(pos);
    if (out.length > count) out.shift();
  }
  if (pos.whiteToMove !== current.whiteToMove || pos.squares.join("") !== current.squares.join("")) {
    throw new Error("maia: history does not lead to the given FEN");
  }
  return out;
}

/**
 * Writes the active one-hot feature indices for each square into `active` (square * 8 + slot)
 * and counts into `counts`. Feature index = historySlot * 12 + channel, channel 0-5 for the
 * side to move (P N B R Q K), 6-11 for the opponent. Short histories are padded at the front
 * with the earliest position.
 */
export function encodeHistory(positions: Position[], historyLen: number, active: Int32Array, counts: Int32Array): void {
  counts.fill(0);
  const pad = historyLen - positions.length;
  for (let slot = 0; slot < historyLen; slot++) {
    const pos = positions[Math.max(0, slot - pad)]!;
    const flip = !pos.whiteToMove;
    for (let s = 0; s < 64; s++) {
      const piece = pos.squares[s]!;
      if (piece === ".") continue;
      const isWhite = piece === piece.toUpperCase();
      const own = isWhite !== flip;
      const sq = flip ? s ^ 56 : s;
      const feature = slot * 12 + PIECE_CHANNEL[piece.toLowerCase()]! + (own ? 0 : 6);
      active[sq * historyLen + counts[sq]!] = feature;
      counts[sq]!++;
    }
  }
}

/** Mirrors a square index vertically (a1 <-> a8). */
export function mirrorSquare(sq: number): number {
  return sq ^ 56;
}
