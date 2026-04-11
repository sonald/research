"""Tetromino definitions and wall-kick tables."""

from __future__ import annotations

from typing import Final


PIECE_NAMES: Final[tuple[str, ...]] = ("I", "J", "L", "O", "S", "T", "Z")
PIECE_TO_INDEX: Final[dict[str, int]] = {name: index for index, name in enumerate(PIECE_NAMES)}
EMPTY_HOLD_INDEX: Final[int] = len(PIECE_NAMES)

SPAWN_MATRICES: Final[dict[str, tuple[str, str, str, str]]] = {
    "I": ("....", "IIII", "....", "...."),
    "J": ("....", "J...", "JJJ.", "...."),
    "L": ("....", "..L.", "LLL.", "...."),
    "O": ("....", ".OO.", ".OO.", "...."),
    "S": ("....", ".SS.", "SS..", "...."),
    "T": ("....", ".T..", "TTT.", "...."),
    "Z": ("....", "ZZ..", ".ZZ.", "...."),
}

JLSTZ_KICKS: Final[dict[tuple[int, int], tuple[tuple[int, int], ...]]] = {
    (0, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (1, 0): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (1, 2): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (2, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (2, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (3, 2): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (3, 0): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (0, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
}

I_KICKS: Final[dict[tuple[int, int], tuple[tuple[int, int], ...]]] = {
    (0, 1): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
    (1, 0): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
    (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
    (2, 1): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
    (2, 3): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
    (3, 2): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
    (3, 0): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
    (0, 3): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
}


def _rotate_matrix_cw(matrix: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
    size = len(matrix)
    rotated = []
    for row in range(size):
        rotated.append("".join(matrix[size - 1 - col][row] for col in range(size)))
    return tuple(rotated)  # type: ignore[return-value]


def build_piece_states() -> dict[str, tuple[tuple[tuple[int, int], ...], ...]]:
    states: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {}
    for name, spawn in SPAWN_MATRICES.items():
        if name == "O":
            cells = tuple(
                (row, col)
                for row, row_text in enumerate(spawn)
                for col, cell in enumerate(row_text)
                if cell != "."
            )
            states[name] = (cells, cells, cells, cells)
            continue

        rotations = [spawn]
        for _ in range(3):
            rotations.append(_rotate_matrix_cw(rotations[-1]))

        state_cells: list[tuple[tuple[int, int], ...]] = []
        for rotation in rotations:
            cells = tuple(
                (row, col)
                for row, row_text in enumerate(rotation)
                for col, cell in enumerate(row_text)
                if cell != "."
            )
            state_cells.append(cells)
        states[name] = tuple(state_cells)
    return states


PIECE_STATES: Final[dict[str, tuple[tuple[tuple[int, int], ...], ...]]] = build_piece_states()


def get_kicks(piece_name: str, from_rotation: int, to_rotation: int) -> tuple[tuple[int, int], ...]:
    if piece_name == "O":
        return ((0, 0),)
    if piece_name == "I":
        return I_KICKS[(from_rotation, to_rotation)]
    return JLSTZ_KICKS[(from_rotation, to_rotation)]
