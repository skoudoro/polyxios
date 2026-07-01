"""Medit .meshb binary codec (GmFlib format) — read + write."""

import mmap
from pathlib import Path
import struct
from typing import Any

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".meshb"

# GmFlib keyword codes
_KW_VERSION = 1
_KW_DIMENSION = 3
_KW_VERTICES = 4
# _KW_EDGES (5) is in elem_rec_sizes so scanning can advance past edge sections,
# but edges are not decoded or written — they carry no PolyData element type.
_KW_EDGES = 5
_KW_TRIANGLES = 6
_KW_QUADRILATERALS = 7
_KW_TETRAHEDRA = 8
_KW_HEXAHEDRA = 10
_KW_END = 54

# Keyword → (polyxios element name, nodes per element)
_KW_TO_ELEM: dict[int, tuple[str, int]] = {
    _KW_TRIANGLES: ("triangle", 3),
    _KW_QUADRILATERALS: ("quad", 4),
    _KW_TETRAHEDRA: ("tetra", 4),
    _KW_HEXAHEDRA: ("hexahedron", 8),
}
_ELEM_TO_KW: dict[str, int] = {
    "triangle": _KW_TRIANGLES,
    "quad": _KW_QUADRILATERALS,
    "tetra": _KW_TETRAHEDRA,
    "hexahedron": _KW_HEXAHEDRA,
}


def read(path: Path | str) -> PolyData:
    """Parse a Medit binary mesh file (.meshb) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .meshb file.

    Returns
    -------
    PolyData
        Parsed mesh. Element reference integers are stored in
        element_attrs["ref"] as int32.

    Raises
    ------
    CodecError
        On unrecognised magic number, unsupported version, truncated data,
        or unknown GmFlib keyword that prevents section scanning.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            return _decode(mm)
        finally:
            mm.close()


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise PolyData to a Medit binary mesh file (.meshb).

    Writes version 2 (float64, 32-bit counts). If element_attrs["ref"] is
    present, those integers are written as section reference tags; otherwise
    reference tags default to 0.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output .meshb file path.
    """
    path = Path(path)
    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)
    dim = poly.vertices.shape[1]

    # Group elements by type to emit sections
    groups: dict[int, list[int]] = {}
    for i in range(n_elems):
        name = ELEMENT_TYPES_INV.get(int(poly.element_types[i]), "")
        kw = _ELEM_TO_KW.get(name)
        if kw is not None:
            groups.setdefault(kw, []).append(i)

    with open(path, "wb") as fh:
        _wi32 = lambda v: fh.write(struct.pack("<i", v))  # noqa: E731

        # File header
        _wi32(_KW_VERSION)
        _wi32(2)  # version 2 = float64
        _wi32(_KW_DIMENSION)
        _wi32(dim)

        # Vertices section
        _wi32(_KW_VERTICES)
        _wi32(n_verts)
        vert_dt = np.dtype([("xyz", "<f8", (dim,)), ("ref", "<i4")])
        buf = np.zeros(n_verts, dtype=vert_dt)
        buf["xyz"] = poly.vertices[:, :dim]
        fh.write(buf.tobytes())

        # Element sections — vectorised gather per type
        refs_attr = poly.element_attrs.get("ref")
        for kw, indices in groups.items():
            _, n_nodes = _KW_TO_ELEM[kw]
            _wi32(kw)
            _wi32(len(indices))
            idx = np.asarray(indices, dtype=np.intp)
            starts = np.asarray(poly.offsets[idx], dtype=np.intp)
            flat_idx = (starts[:, None] + np.arange(n_nodes, dtype=np.intp)).ravel()
            # Medit uses 1-based indices
            nodes = (
                poly.connectivity[flat_idx].reshape(len(indices), n_nodes) + 1
            ).astype(np.int32)
            if refs_attr is not None:
                refs_col = refs_attr[idx].reshape(-1, 1).astype(np.int32)
            else:
                refs_col = np.zeros((len(indices), 1), dtype=np.int32)
            block = np.concatenate([nodes, refs_col], axis=1)
            fh.write(block.tobytes())

        _wi32(_KW_END)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_header(mm: mmap.mmap) -> tuple[int, int, str]:
    """Read magic, version, endian from mmap. Returns (version, dim, endian_char)."""
    if len(mm) < 8:
        raise CodecError("File too short for .meshb header.")

    kw = struct.unpack_from("<i", mm, 0)[0]
    if kw == _KW_VERSION:
        endian = "<"
    else:
        kw_be = struct.unpack_from(">i", mm, 0)[0]
        if kw_be == _KW_VERSION:
            endian = ">"
        else:
            raise CodecError(
                f".meshb: first word is {kw} (LE) / {kw_be} (BE); "
                "expected 1 (GmfMeshVersionFormatted)."
            )

    fmt_i = endian + "i"
    version = struct.unpack_from(fmt_i, mm, 4)[0]
    if version not in (1, 2):
        raise CodecError(f".meshb version {version} not supported (expected 1 or 2).")

    # Dimension keyword + value
    kw2 = struct.unpack_from(fmt_i, mm, 8)[0]
    if kw2 != _KW_DIMENSION:
        raise CodecError(
            f".meshb: expected Dimension keyword (3) at offset 8, got {kw2}."
        )
    dim = struct.unpack_from(fmt_i, mm, 12)[0]

    return version, dim, endian


def _scan_sections(
    mm: mmap.mmap, version: int, dim: int, endian: str
) -> dict[int, tuple[int, int]]:
    """Return {keyword: (count, data_byte_offset)} for all known sections."""
    fmt_i = endian + "i"
    pos = 16  # after header (4+4+4+4 bytes)
    sections: dict[int, tuple[int, int]] = {}
    total = len(mm)

    # Record sizes (bytes per record, including trailing ref int32)
    float_size = 4 if version == 1 else 8
    vert_rec = dim * float_size + 4  # x[, y[, z]] + ref
    elem_rec_sizes = {
        _KW_TRIANGLES: 3 * 4 + 4,
        _KW_QUADRILATERALS: 4 * 4 + 4,
        _KW_TETRAHEDRA: 4 * 4 + 4,
        _KW_HEXAHEDRA: 8 * 4 + 4,
        _KW_EDGES: 2 * 4 + 4,
    }

    while pos + 8 <= total:
        kw = struct.unpack_from(fmt_i, mm, pos)[0]
        if kw == _KW_END:
            break
        count = struct.unpack_from(fmt_i, mm, pos + 4)[0]
        data_start = pos + 8
        if kw == _KW_VERTICES:
            rec = vert_rec
        else:
            rec = elem_rec_sizes.get(kw, 0)
        if rec == 0:
            raise CodecError(
                f".meshb: unknown keyword {kw} at file offset {pos}; "
                "cannot determine record size. "
                f"Supported keywords: {sorted(elem_rec_sizes) + [_KW_VERTICES]}."
            )
        sections[kw] = (count, data_start)
        pos = data_start + count * rec

    return sections


def _decode(mm: mmap.mmap) -> PolyData:
    version, dim, endian = _parse_header(mm)
    sections = _scan_sections(mm, version, dim, endian)

    float_dt = endian + ("f4" if version == 1 else "f8")
    float_size = 4 if version == 1 else 8

    # --- Vertices ---
    vertices = np.zeros((0, 3), dtype=np.float64)
    if _KW_VERTICES in sections:
        n_verts, vstart = sections[_KW_VERTICES]
        rec = dim * float_size + 4
        nbytes = n_verts * rec
        vert_dt = np.dtype([("xyz", float_dt, (dim,)), ("ref", endian + "i4")])
        verts_arr = np.frombuffer(bytes(mm[vstart : vstart + nbytes]), dtype=vert_dt)
        xyz = verts_arr["xyz"].astype(np.float64)
        vertices = np.zeros((n_verts, 3), dtype=np.float64)
        vertices[:, :dim] = xyz

    # --- Elements ---
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    refs_list: list[int] = []

    for kw, (elem_name, n_nodes) in _KW_TO_ELEM.items():
        if kw not in sections:
            continue
        n_elems, estart = sections[kw]
        rec = n_nodes * 4 + 4
        nbytes = n_elems * rec
        elem_dt = np.dtype(
            [("nodes", endian + "i4", (n_nodes,)), ("ref", endian + "i4")]
        )
        raw = bytes(mm[estart : estart + nbytes])
        arr = np.frombuffer(raw, dtype=elem_dt)
        # Convert 1-based to 0-based
        conn_2d = arr["nodes"].astype(np.int32) - 1
        refs = arr["ref"].astype(np.int32)
        conn_list.extend(conn_2d.ravel().tolist())
        base = offsets_list[-1]
        offsets_list.extend((base + np.arange(1, n_elems + 1) * n_nodes).tolist())
        code = ELEMENT_TYPES.get(elem_name, 0)
        types_list.extend([code] * n_elems)
        refs_list.extend(refs.tolist())

    elem_attrs: dict[str, np.ndarray] = {}
    if refs_list:
        elem_attrs["ref"] = np.array(refs_list, dtype=np.int32)

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        element_attrs=elem_attrs,
    )
