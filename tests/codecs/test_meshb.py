from __future__ import annotations

import tempfile

import numpy as np

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


def test_roundtrip_tetra() -> None:
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".meshb", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_triangles() -> None:
    poly = _tri_mesh()
    with tempfile.NamedTemporaryFile(suffix=".meshb", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 4
    np.testing.assert_allclose(poly2.vertices, poly.vertices)


def test_binary_file_has_correct_magic() -> None:
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".meshb", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    import struct

    with open(tmp, "rb") as f:
        raw = f.read(8)
    kw, version = struct.unpack("<ii", raw)
    assert kw == 1  # GmfMeshVersionFormatted
    assert version == 2  # float64


def test_lazy_roundtrip() -> None:
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".meshb", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly_lazy = read(tmp, lazy=True)
    assert len(poly_lazy.element_types) == 1
    np.testing.assert_allclose(poly_lazy.vertices, poly.vertices)
    np.testing.assert_array_equal(poly_lazy.connectivity, poly.connectivity)


def test_ref_stored_in_element_attrs() -> None:
    poly = _tri_mesh()
    with tempfile.NamedTemporaryFile(suffix=".meshb", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert "ref" in poly2.element_attrs
    assert len(poly2.element_attrs["ref"]) == 4


def test_mixed_elements() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
    )
    with tempfile.NamedTemporaryFile(suffix=".meshb", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 2
