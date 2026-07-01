from __future__ import annotations

import struct

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._meshb import read, write


def _tet_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))],
    )


def test_roundtrip_tetra(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_triangles(tmp_path) -> None:
    poly = _tri_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 4
    np.testing.assert_allclose(poly2.vertices, poly.vertices)


def test_binary_file_has_correct_magic(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    with open(tmp, "rb") as f:
        raw = f.read(8)
    kw, version = struct.unpack("<ii", raw)
    assert kw == 1  # GmfMeshVersionFormatted
    assert version == 2  # float64


def test_ref_stored_in_element_attrs(tmp_path) -> None:
    poly = _tri_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    poly2 = read(tmp)
    assert "ref" in poly2.element_attrs
    assert len(poly2.element_attrs["ref"]) == 4


def test_ref_roundtrip(tmp_path) -> None:
    poly = _tri_mesh()
    refs = np.array([10, 20, 30, 40], dtype=np.int32)
    import dataclasses

    poly = dataclasses.replace(poly, element_attrs={"ref": refs})
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_array_equal(poly2.element_attrs["ref"], refs)


def test_mixed_elements(tmp_path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
    )
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 2


def test_unknown_keyword_raises(tmp_path) -> None:
    """File with unknown GmFlib keyword before element data raises CodecError."""
    import struct as st

    from polyxios.exceptions import CodecError

    poly = _tet_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    data = tmp.read_bytes()
    # Inject unknown keyword 99 (count=0) just after the header (offset 16)
    injected = data[:16] + st.pack("<ii", 99, 0) + data[16:]
    bad = tmp_path / "bad.meshb"
    bad.write_bytes(injected)
    with pytest.raises(CodecError, match="unknown keyword 99"):
        read(bad)
