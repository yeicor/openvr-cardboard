"""ArUco dictionary helpers, including the low-density 3x3 board set.

The project ships a custom 3x3 dictionary instead of OpenCV's smallest 4x4
dictionary.  A 3x3 marker has one module less per side, so every module is
larger on the printed board and survives motion blur and distance better.
Fifteen codes are enough for the HMD and the two hand boards; body tracking is
done by MediaPipe and does not consume marker IDs.

The bit patterns are embedded here rather than produced by
``cv2.aruco.extendDictionary`` at runtime so the printed sheets and every
desktop build agree regardless of the installed OpenCV version.
"""

from __future__ import annotations

import numpy as np

from .config import CUSTOM_DICTIONARY_3X3

# Fifteen 3x3 bit patterns.  The set was chosen to maximise the minimum
# Hamming distance across all four marker rotations; the minimum is two, so a
# single flipped module can never decode as a different code.
_3X3_CODES = (
    "011011111",
    "000010010",
    "111010101",
    "111011000",
    "010110010",
    "100010011",
    "001000011",
    "100101101",
    "011111011",
    "111110011",
    "010100000",
    "110011100",
    "010101001",
    "001001101",
    "011001011",
)

MARKER_SIZE = 3


def _bits(text: str) -> tuple[int, ...]:
    return tuple(1 if value == "1" else 0 for value in text)


def _rotate(bits: tuple[int, ...]) -> tuple[int, ...]:
    """Rotate a row-major 3x3 bit pattern 90 degrees counter-clockwise.

    OpenCV's dictionary stores rotations in counter-clockwise order
    (0, 90, 180, 270).  Using the opposite direction swaps the 90 and 270
    entries; ``identify`` then reports a 270-degree image rotation for a
    marker seen at 90 degrees, so PnP receives corners in the wrong order.
    """
    return tuple(bits[column * 3 + (2 - row)] for row in range(3) for column in range(3))


def _rotations(bits: tuple[int, ...]) -> list[tuple[int, ...]]:
    result = []
    current = bits
    for _ in range(4):
        result.append(current)
        current = _rotate(current)
    return result


def _pack(rotation: tuple[int, ...]) -> np.ndarray:
    """Pack nine bits into OpenCV's big-endian two-byte ArUco row format.

    OpenCV stores the first eight bits in the high byte and the ninth bit in
    the low bit of the second byte (verified against
    ``aruco.Dictionary_getByteListFromBits``).  A 3x3 marker is the only case
    where the final byte is not fully used, so getting this wrong silently
    zeroes the ninth payload module.
    """
    value = 0
    for bit in rotation:
        value = (value << 1) | int(bit)
    return np.asarray([(value >> 1) & 0xFF, value & 0x01], dtype=np.uint8)


def _bytes_list() -> np.ndarray:
    """Build OpenCV's ``(markers, bytes, rotations)`` byte list.

    OpenCV stores the four rotations' bytes consecutively and reshapes the
    result to ``(markers, nBytes, 4)``; for 3x3 markers that means two bytes
    per rotation, interleaved.  Keeping the same order here matters: a
    different order makes ``identify`` match a marker at 0 degrees to its
    180-degree entry, which flips the PnP corner order.
    """
    rows = []
    for text in _3X3_CODES:
        rotations = _rotations(_bits(text))
        rows.append(np.concatenate([_pack(rotation) for rotation in rotations]))
    return np.asarray(rows, dtype=np.uint8).reshape(len(_3X3_CODES), 2, 4)


def get_dictionary(cv2, name: str):
    """Return an OpenCV ArUco dictionary for a built-in or custom name."""
    name = str(name).strip().upper()
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise ValueError("this OpenCV build has no aruco module")
    if name == CUSTOM_DICTIONARY_3X3:
        constructor = getattr(aruco, "Dictionary", None)
        if constructor is None:
            raise ValueError("this OpenCV build cannot load a custom ArUco dictionary")
        # maxCorrectionBits=0: with fifteen 3x3 codes the minimum
        # rotation-invariant Hamming distance is two, so error correction is
        # not safe. Corner refinement still gives sub-pixel corner positions.
        return constructor(_bytes_list(), MARKER_SIZE, 0)
    attribute = getattr(aruco, name, None)
    if attribute is None:
        raise ValueError(f"unknown ArUco dictionary {name!r}")
    getter = getattr(aruco, "getPredefinedDictionary", None)
    if getter is None:  # OpenCV < 4.7
        return aruco.Dictionary_get(attribute)
    return getter(attribute)


def marker_image(cv2, dictionary, marker_id: int, pixels: int):
    """Render one marker, including its one-module black border."""
    generator = getattr(cv2.aruco, "generateImageMarker", None)
    if generator is not None:
        return generator(dictionary, int(marker_id), int(pixels))
    return cv2.aruco.drawMarker(dictionary, int(marker_id), int(pixels))
