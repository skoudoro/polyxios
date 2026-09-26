"""COLLADA: the XML scene format, read and written as a SceneData.

Every file is spelled inline by the builders at the top so a test names
exactly the element it exercises; the reference-implementation checks at
the bottom need pycollada and skip without it.

The two reference-implementation tests silence warnings wholesale: pycollada
0.8 sets an array's shape in place and uses np.matrix, both of which numpy 2.5
warns about on every call.
"""

from __future__ import annotations

import base64
import dataclasses
import io
from pathlib import Path
import re
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import polyxios
from polyxios import PolyData, SceneData, SceneMaterial, SceneNode, make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._scene import SceneImage, SceneTexture
from polyxios.codecs import _collada
from polyxios.codecs._collada import read, read_scene, write, write_scene
from polyxios.exceptions import CodecError, LazyReadError

_NS = "http://www.collada.org/2005/11/COLLADASchema"
_PARAM_TYPE = {"TRANSFORM": "float4x4"}

_TRI = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_QUAD = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])


def _nums(values) -> str:
    return " ".join(str(v) for v in np.asarray(values).ravel().tolist())


def _source(sid: str, values, params: str, *, kind: str = "float_array") -> str:
    arr = np.asarray(values)
    stride = arr.shape[1] if arr.ndim == 2 else 1
    count = arr.shape[0] if arr.ndim == 2 else len(arr)
    text = _nums(values) if kind == "float_array" else " ".join(values)
    names = "".join(
        f'<param name="{p}" type="{_PARAM_TYPE.get(p, "Name" if kind == "Name_array" else "float")}"/>'
        for p in params.split()
    )
    return (
        f'<source id="{sid}"><{kind} id="{sid}-array" count="{count * stride}">'
        f"{text}</{kind}><technique_common>"
        f'<accessor source="#{sid}-array" count="{count}" stride="{stride}">'
        f"{names}</accessor></technique_common></source>"
    )


def _geometry(gid: str, positions, prims: str, *, sources: str = "", name=None) -> str:
    nm = f' name="{name}"' if name else ""
    return (
        f'<geometry id="{gid}"{nm}><mesh>'
        + _source(f"{gid}-pos", positions, "X Y Z")
        + sources
        + f'<vertices id="{gid}-vtx"><input semantic="POSITION" source="#{gid}-pos"/>'
        "</vertices>" + prims + "</mesh></geometry>"
    )


def _triangles(gid: str, faces, *, material=None, extra_inputs: str = "") -> str:
    faces = np.asarray(faces)
    mat = f' material="{material}"' if material else ""
    return (
        f'<triangles count="{len(faces)}"{mat}>'
        f'<input semantic="VERTEX" source="#{gid}-vtx" offset="0"/>'
        + extra_inputs
        + f"<p>{_nums(faces)}</p></triangles>"
    )


def _scene_with(gid: str, *, node_extra: str = "", bind: str = "") -> str:
    return (
        '<library_visual_scenes><visual_scene id="Scene" name="Scene">'
        f'<node id="n0" name="thing">{node_extra}'
        f'<instance_geometry url="#{gid}">{bind}</instance_geometry>'
        "</node></visual_scene></library_visual_scenes>"
        '<scene><instance_visual_scene url="#Scene"/></scene>'
    )


def _dae(body: str, *, version: str = "1.4.1", asset: str = "") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<COLLADA xmlns="{_NS}" version="{version}">'
        f"<asset>{asset}</asset>{body}</COLLADA>\n"
    )


def _tri_dae(**kw) -> str:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    return _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))


def _write(tmp_path: Path, text: str, name: str = "m.dae") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _surface() -> PolyData:
    return make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]]),
        [("triangle", np.array([[0, 1, 4]])), ("quad", np.array([[0, 1, 2, 3]]))],
    )


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_read_triangles(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _tri_dae()))
    assert len(scene.meshes) == 1
    mesh = scene.meshes[0]
    np.testing.assert_array_equal(mesh.vertices, _TRI)
    np.testing.assert_array_equal(mesh.connectivity, [0, 1, 2])
    assert mesh.element_types.tolist() == [ELEMENT_TYPES["triangle"]]
    assert scene.nodes[0].mesh == 0
    assert scene.nodes[0].name == "thing"
    assert scene.scenes == ((0,),)
    assert scene.active_scene == 0
    assert "material" not in mesh.element_attrs


def test_read_polylist_quads_and_polygons(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]])
    prim = (
        '<polylist count="2"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<vcount>4 5</vcount><p>0 1 2 3 1 4 5 2 3</p></polylist>"
    )
    geo = _geometry("g0", verts, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert mesh.element_types.tolist() == [
        ELEMENT_TYPES["quad"],
        ELEMENT_TYPES["polygon"],
    ]
    assert mesh.offsets.tolist() == [0, 4, 9]
    assert mesh.connectivity.tolist() == [0, 1, 2, 3, 1, 4, 5, 2, 3]


def test_read_polylist_triangles_are_triangles(tmp_path: Path) -> None:
    prim = (
        '<polylist count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<vcount>3</vcount><p>0 1 2</p></polylist>"
    )
    geo = _geometry("g0", _TRI, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert mesh.element_types.tolist() == [ELEMENT_TYPES["triangle"]]


def test_read_polygons_element_with_holes_warns(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]])
    prim = (
        '<polygons count="2"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<p>0 1 2 3</p><ph><p>0 1 4</p><h>1 2 4</h></ph></polygons>"
    )
    geo = _geometry("g0", verts, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.warns(UserWarning, match="hole"):
        mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert mesh.element_types.tolist() == [
        ELEMENT_TYPES["quad"],
        ELEMENT_TYPES["triangle"],
    ]
    assert mesh.connectivity.tolist() == [0, 1, 2, 3, 0, 1, 4]


def test_read_lines_and_linestrips(tmp_path: Path) -> None:
    prim = (
        '<lines count="2"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<p>0 1 1 2</p></lines>"
        '<linestrips count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<p>0 1 2</p></linestrips>"
    )
    geo = _geometry("g0", _TRI, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert mesh.element_types.tolist() == [
        ELEMENT_TYPES["line"],
        ELEMENT_TYPES["line"],
        ELEMENT_TYPES["poly_line"],
    ]
    assert mesh.connectivity.tolist() == [0, 1, 1, 2, 0, 1, 2]


def test_read_tristrips_and_trifans(tmp_path: Path) -> None:
    prim = (
        '<tristrips count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<p>0 1 2 3</p></tristrips>"
        '<trifans count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<p>0 1 2 3</p></trifans>"
    )
    geo = _geometry("g0", _QUAD, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert mesh.element_types.tolist() == [
        ELEMENT_TYPES["triangle_strip"],
        ELEMENT_TYPES["triangle"],
        ELEMENT_TYPES["triangle"],
    ]
    assert mesh.connectivity.tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 2, 3]


def test_read_multi_offset_inputs_split_corners(tmp_path: Path) -> None:
    normals = np.array([[0.0, 0, 1], [0, 0, -1]])
    uv = np.array([[0.0, 0], [1, 0], [0, 1], [1, 1]])
    faces = np.array(
        [[0, 0, 0, 1, 0, 1, 2, 0, 2], [0, 1, 3, 1, 0, 1, 2, 0, 2]], dtype=np.int64
    )
    inputs = (
        '<input semantic="NORMAL" source="#g0-nrm" offset="1"/>'
        '<input semantic="TEXCOORD" source="#g0-uv" offset="2" set="0"/>'
    )
    geo = _geometry(
        "g0",
        _TRI,
        _triangles("g0", faces, extra_inputs=inputs),
        sources=_source("g0-nrm", normals, "X Y Z") + _source("g0-uv", uv, "S T"),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    # Corner (0, 0, 0) and (0, 1, 3) differ, so vertex 0 is split; the other
    # two corners agree across faces and stay one vertex each.
    assert len(mesh.vertices) == 4
    np.testing.assert_array_equal(mesh.vertices, np.vstack([_TRI, _TRI[:1]]))
    assert mesh.connectivity.tolist() == [0, 1, 2, 3, 1, 2]
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"], normals[[0, 0, 0, 1]])
    np.testing.assert_array_equal(mesh.vertex_attrs["texcoords"], uv[[0, 1, 2, 3]])


def test_read_shared_offset_keeps_unreferenced_vertices(tmp_path: Path) -> None:
    verts = np.vstack([_TRI, [[5.0, 5, 5]]])
    normals = np.tile([[0.0, 0, 1]], (4, 1))
    inputs = '<input semantic="NORMAL" source="#g0-nrm" offset="0"/>'
    geo = _geometry(
        "g0",
        verts,
        _triangles("g0", [[0, 1, 2]], extra_inputs=inputs),
        sources=_source("g0-nrm", normals, "X Y Z"),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert len(mesh.vertices) == 4
    assert mesh.vertex_attrs["normals"].shape == (4, 3)


def test_read_inputs_inside_vertices(tmp_path: Path) -> None:
    normals = np.tile([[0.0, 0, 1]], (3, 1))
    geo = (
        '<geometry id="g0"><mesh>'
        + _source("g0-pos", _TRI, "X Y Z")
        + _source("g0-nrm", normals, "X Y Z")
        + '<vertices id="g0-vtx"><input semantic="POSITION" source="#g0-pos"/>'
        '<input semantic="NORMAL" source="#g0-nrm"/></vertices>'
        + _triangles("g0", [[0, 1, 2]])
        + "</mesh></geometry>"
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"], normals)


def test_read_colors_and_texcoord_sets(tmp_path: Path) -> None:
    colors = np.array([[1.0, 0, 0, 1], [0, 1, 0, 0.5], [0, 0, 1, 1]])
    uv1 = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    inputs = (
        '<input semantic="COLOR" source="#g0-col" offset="0" set="0"/>'
        '<input semantic="TEXCOORD" source="#g0-uv1" offset="0" set="1"/>'
        '<input semantic="TEXTANGENT" source="#g0-tan" offset="0"/>'
    )
    geo = _geometry(
        "g0",
        _TRI,
        _triangles("g0", [[0, 1, 2]], extra_inputs=inputs),
        sources=_source("g0-col", colors, "R G B A")
        + _source("g0-uv1", uv1, "S T")
        + _source("g0-tan", np.tile([[1.0, 0, 0]], (3, 1)), "X Y Z"),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    np.testing.assert_array_equal(mesh.vertex_attrs["colors"], colors)
    np.testing.assert_array_equal(mesh.vertex_attrs["texcoords_1"], uv1)
    assert mesh.vertex_attrs["textangent"].shape == (3, 3)


def test_read_accessor_offset_and_unnamed_params(tmp_path: Path) -> None:
    src = (
        '<source id="g0-pos"><float_array id="g0-pos-array" count="13">'
        "9 0 0 0 7 1 0 0 7 0 1 0 7</float_array><technique_common>"
        '<accessor source="#g0-pos-array" count="3" stride="4" offset="1">'
        '<param name="X" type="float"/><param name="Y" type="float"/>'
        '<param name="Z" type="float"/><param type="float"/></accessor>'
        "</technique_common></source>"
    )
    geo = (
        '<geometry id="g0"><mesh>'
        + src
        + '<vertices id="g0-vtx"><input semantic="POSITION" source="#g0-pos"/>'
        "</vertices>" + _triangles("g0", [[0, 1, 2]]) + "</mesh></geometry>"
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    np.testing.assert_array_equal(mesh.vertices, _TRI)


def test_read_geometry_never_instanced_still_a_mesh(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]])) + _geometry(
        "g1", _QUAD, _triangles("g1", [[0, 1, 2]])
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g1"))
    scene = read_scene(_write(tmp_path, text))
    assert len(scene.meshes) == 2
    assert scene.nodes[0].mesh == 1


def test_read_geometry_without_mesh_warns(tmp_path: Path) -> None:
    geo = '<geometry id="g0"><spline/></geometry>'
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.warns(UserWarning, match="spline"):
        scene = read_scene(_write(tmp_path, text))
    assert len(scene.meshes[0].vertices) == 0


def test_read_vertices_input_of_wrong_length_refused(tmp_path: Path) -> None:
    normals = np.tile([[0.0, 0, 1]], (2, 1))
    geo = (
        '<geometry id="g0"><mesh>'
        + _source("g0-pos", _TRI, "X Y Z")
        + _source("g0-nrm", normals, "X Y Z")
        + '<vertices id="g0-vtx"><input semantic="NORMAL" source="#g0-nrm"/>'
        '<input semantic="POSITION" source="#g0-pos"/></vertices>'
        + _triangles("g0", [[0, 1, 2]])
        + "</mesh></geometry>"
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.raises(CodecError, match="normals inside <vertices> with 2 rows"):
        read_scene(_write(tmp_path, text))


def test_read_shared_offset_short_source_refused_past_its_end(tmp_path: Path) -> None:
    normals = np.tile([[0.0, 0, 1]], (2, 1))
    inputs = '<input semantic="NORMAL" source="#g0-nrm" offset="0"/>'
    geo = _geometry(
        "g0",
        _TRI,
        _triangles("g0", [[0, 1, 2]], extra_inputs=inputs),
        sources=_source("g0-nrm", normals, "X Y Z"),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.raises(CodecError, match="entry 2 of a source holding 2"):
        read_scene(_write(tmp_path, text))


def test_read_accessor_stride_past_its_named_params(tmp_path: Path) -> None:
    src = (
        '<source id="g0-pos"><float_array id="g0-pos-array" count="12">'
        "0 0 0 7 1 0 0 7 0 1 0 7</float_array><technique_common>"
        '<accessor source="#g0-pos-array" count="3" stride="4">'
        '<param name="X" type="float"/><param name="Y" type="float"/>'
        '<param name="Z" type="float"/></accessor></technique_common></source>'
    )
    geo = (
        '<geometry id="g0"><mesh>'
        + src
        + '<vertices id="g0-vtx"><input semantic="POSITION" source="#g0-pos"/>'
        "</vertices>" + _triangles("g0", [[0, 1, 2]]) + "</mesh></geometry>"
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    np.testing.assert_array_equal(mesh.vertices, _TRI)


def test_read_block_without_an_input_gets_zeros(tmp_path: Path) -> None:
    normals = np.array([[0.0, 0, 1], [0, 0, -1], [0, 1, 0]])
    inputs = '<input semantic="NORMAL" source="#g0-nrm" offset="1"/>'
    prims = _triangles("g0", [[0, 0, 1, 1, 2, 2]], extra_inputs=inputs) + (
        '<lines count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<p>0 1</p></lines>"
    )
    geo = _geometry("g0", _TRI, prims, sources=_source("g0-nrm", normals, "X Y Z"))
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.warns(UserWarning, match="normals.*some primitive blocks"):
        mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert len(mesh.vertices) == 5
    n = mesh.vertex_attrs["normals"]
    assert np.isfinite(n).all()
    np.testing.assert_array_equal(n[:3], normals)
    np.testing.assert_array_equal(n[3:], 0.0)


def test_read_attr_of_two_widths_warns_once(tmp_path: Path) -> None:
    n3 = np.array([[0.0, 0, 1], [0, 0, -1], [0, 1, 0]])
    n2 = np.array([[1.0, 0], [0, 1]])
    tri = _triangles(
        "g0",
        [[0, 0, 1, 1, 2, 2]],
        extra_inputs='<input semantic="NORMAL" source="#g0-n3" offset="1"/>',
    )
    lines = (
        '<lines count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        '<input semantic="NORMAL" source="#g0-n2" offset="1"/><p>0 0 1 1</p></lines>'
    )
    sources = _source("g0-n3", n3, "X Y Z") + _source("g0-n2", n2, "X Y")
    geo = _geometry("g0", _TRI, tri + lines, sources=sources)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mesh = read_scene(_write(tmp_path, text)).meshes[0]
    messages = [str(w.message) for w in record]
    assert len(messages) == 1 and "two widths" in messages[0]
    assert mesh.vertex_attrs["normals"].shape == (5, 3)


def test_read_symbols_resolved_per_block(tmp_path: Path) -> None:
    prims = (
        _triangles("g0", [[0, 1, 2]] * 3, material="sym")
        + _triangles("g0", [[0, 1, 2]] * 2, material="ghost")
        + _triangles("g0", [[0, 1, 2]] * 2)
    )
    text = _dae(
        f"<library_geometries>{_geometry('g0', _TRI, prims)}</library_geometries>"
        '<library_materials><material id="sym"/></library_materials>'
        + _scene_with("g0")
    )
    path = _write(tmp_path, text)
    with pytest.warns(UserWarning, match=r"\['ghost'\]"):
        mesh = read_scene(path).meshes[0]
    assert mesh.element_attrs["material"].tolist() == [0, 0, 0, -1, -1, -1, -1]
    root = ET.fromstring(path.read_bytes())
    doc = _collada._Doc(
        root=root, name="m.dae", size=len(text), ids=_collada._ids(root)
    )
    geo = root.find(f"{{{_NS}}}library_geometries/{{{_NS}}}geometry")
    assert _collada._read_geometry(doc, geo, 0).symbols == [
        ("sym", 3),
        ("ghost", 2),
        (None, 2),
    ]


def test_read_per_position_source_reindexed_by_a_later_block(tmp_path: Path) -> None:
    normals = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    tri = _triangles(
        "g0",
        [[0, 1, 2]],
        extra_inputs='<input semantic="NORMAL" source="#g0-nrm" offset="0"/>',
    )
    lines = (
        '<lines count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        '<input semantic="NORMAL" source="#g0-nrm" offset="1"/><p>0 2 1 2</p></lines>'
    )
    geo = _geometry(
        "g0", _TRI, tri + lines, sources=_source("g0-nrm", normals, "X Y Z")
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    # The line corners name normal 2 for positions 0 and 1, so they split.
    assert len(mesh.vertices) == 5
    assert mesh.connectivity.tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"][:3], normals)
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"][3:], normals[[2, 2]])


def test_read_second_per_position_source_of_one_attr_splits_its_corners(
    tmp_path: Path,
) -> None:
    n1 = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    n2 = -n1
    tri = _triangles(
        "g0",
        [[0, 1, 2]],
        extra_inputs='<input semantic="NORMAL" source="#g0-n1" offset="0"/>',
    )
    lines = (
        '<lines count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        '<input semantic="NORMAL" source="#g0-n2" offset="0"/><p>0 1</p></lines>'
    )
    sources = _source("g0-n1", n1, "X Y Z") + _source("g0-n2", n2, "X Y Z")
    geo = _geometry("g0", _TRI, tri + lines, sources=sources)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    # The line names the second source for positions 0 and 1, so those two
    # corners are new vertices carrying its values, not the first source's.
    assert len(mesh.vertices) == 5
    assert mesh.connectivity.tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"][:3], n1)
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"][3:], n2[:2])


def test_read_geometries_sharing_a_source_own_their_memory(tmp_path: Path) -> None:
    normals = np.tile([[0.0, 0, 1]], (3, 1))
    shared = _source("shared-pos", _TRI, "X Y Z") + _source(
        "shared-n", normals, "X Y Z"
    )
    geos = "".join(
        f'<geometry id="{gid}"><mesh><vertices id="{gid}-vtx">'
        '<input semantic="POSITION" source="#shared-pos"/>'
        '<input semantic="NORMAL" source="#shared-n"/></vertices>'
        + _triangles(gid, [[0, 1, 2]])
        + "</mesh></geometry>"
        for gid in ("g0", "g1")
    )
    text = _dae(
        f"<library_geometries>{shared}{geos}</library_geometries>" + _scene_with("g0")
    )
    a, b = read_scene(_write(tmp_path, text)).meshes
    assert not np.shares_memory(a.vertices, b.vertices)
    assert not np.shares_memory(a.vertex_attrs["normals"], b.vertex_attrs["normals"])
    a.vertices[0] = 9.0
    np.testing.assert_array_equal(b.vertices, _TRI)


# ---------------------------------------------------------------------------
# materials, effects, images
# ---------------------------------------------------------------------------


def _effect_dae(technique: str, *, newparams: str = "", images: str = "") -> str:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]], material="sym"))
    bind = (
        "<bind_material><technique_common>"
        '<instance_material symbol="sym" target="#mat0"/>'
        "</technique_common></bind_material>"
    )
    return _dae(
        f"<library_images>{images}</library_images>"
        f'<library_effects><effect id="fx0"><profile_COMMON>{newparams}'
        f'<technique sid="common">{technique}</technique></profile_COMMON>'
        "</effect></library_effects>"
        '<library_materials><material id="mat0" name="Red">'
        '<instance_effect url="#fx0"/></material></library_materials>'
        f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0", bind=bind)
    )


def test_read_phong_material(tmp_path: Path) -> None:
    technique = (
        "<phong><emission><color>0.1 0.2 0.3 1</color></emission>"
        "<ambient><color>0 0 0 1</color></ambient>"
        "<diffuse><color>1 0 0 1</color></diffuse>"
        "<specular><color>0.5 0.5 0.5 1</color></specular>"
        "<shininess><float>50</float></shininess>"
        "<transparency><float>1</float></transparency></phong>"
    )
    scene = read_scene(_write(tmp_path, _effect_dae(technique)))
    assert len(scene.materials) == 1
    mat = scene.materials[0]
    assert mat.name == "Red"
    assert mat.base_color == (1.0, 0.0, 0.0, 1.0)
    assert mat.emissive == pytest.approx((0.1, 0.2, 0.3))
    assert mat.alpha_mode == "OPAQUE"
    assert mat.metallic == 0.0
    assert mat.extras["shading"] == "phong"
    assert mat.extras["specular"] == (0.5, 0.5, 0.5, 1.0)
    assert mat.extras["shininess"] == 50.0
    assert scene.meshes[0].element_attrs["material"].tolist() == [0]


def test_read_transparency_a_one_becomes_blend(tmp_path: Path) -> None:
    technique = (
        "<lambert><diffuse><color>0 1 0 1</color></diffuse>"
        '<transparent opaque="A_ONE"><color>0 0 0 1</color></transparent>'
        "<transparency><float>0.25</float></transparency></lambert>"
    )
    mat = read_scene(_write(tmp_path, _effect_dae(technique))).materials[0]
    assert mat.base_color == (0.0, 1.0, 0.0, 0.25)
    assert mat.alpha_mode == "BLEND"
    assert mat.extras["shading"] == "lambert"


def test_read_transparency_rgb_zero(tmp_path: Path) -> None:
    technique = (
        "<blinn><diffuse><color>0 1 0 1</color></diffuse>"
        '<transparent opaque="RGB_ZERO"><color>1 1 1 1</color></transparent>'
        "<transparency><float>1</float></transparency></blinn>"
    )
    mat = read_scene(_write(tmp_path, _effect_dae(technique))).materials[0]
    assert mat.base_color[3] == pytest.approx(0.0)
    assert mat.alpha_mode == "BLEND"


def test_read_unknown_opaque_mode_is_a_one_with_a_warning(tmp_path: Path) -> None:
    technique = (
        "<phong><diffuse><color>0 1 0 0.5</color></diffuse>"
        '<transparent opaque="a_one"><color>1 1 1 1</color></transparent>'
        "<transparency><float>1</float></transparency></phong>"
    )
    with pytest.warns(UserWarning, match="opaque='a_one'"):
        mat = read_scene(_write(tmp_path, _effect_dae(technique))).materials[0]
    # The diffuse alpha is applied once, not squared into 0.25.
    assert mat.base_color[3] == pytest.approx(0.5)
    assert mat.extras["transparent_mode"] == "A_ONE"


def test_read_texture_chain_and_images(tmp_path: Path) -> None:
    newparams = (
        '<newparam sid="surf"><surface type="2D"><init_from>img0</init_from>'
        "</surface></newparam>"
        '<newparam sid="samp"><sampler2D><source>surf</source>'
        "<wrap_s>CLAMP</wrap_s><wrap_t>MIRROR</wrap_t>"
        "<minfilter>LINEAR_MIPMAP_LINEAR</minfilter><magfilter>NEAREST</magfilter>"
        "</sampler2D></newparam>"
    )
    technique = (
        '<phong><diffuse><texture texture="samp" texcoord="UVMap"/></diffuse>'
        '<extra><technique profile="FCOLLADA"><bump>'
        '<texture texture="img1" texcoord="UVMap"/></bump></technique></extra>'
        "</phong>"
    )
    images = (
        '<image id="img0" name="skin"><init_from>textures/skin.png</init_from></image>'
        '<image id="img1"><init_from>bump.jpg</init_from></image>'
    )
    scene = read_scene(
        _write(tmp_path, _effect_dae(technique, newparams=newparams, images=images))
    )
    assert scene.images == (
        SceneImage(uri="textures/skin.png", media_type="image/png", name="skin"),
        SceneImage(uri="bump.jpg", media_type="image/jpeg"),
    )
    mat = scene.materials[0]
    assert mat.base_color_texture == 0
    assert mat.normal_texture == 1
    tex = scene.textures[0]
    assert tex.image == 0
    assert (tex.wrap_s, tex.wrap_t) == (33071, 33648)
    assert (tex.min_filter, tex.mag_filter) == (9987, 9728)
    assert scene.textures[1] == SceneTexture(image=1)


def test_warnings_point_at_the_caller(tmp_path: Path) -> None:
    technique = '<phong><diffuse><texture texture="nowhere" texcoord="UVMap"/></diffuse></phong>'
    with pytest.warns(UserWarning, match="reaches no image") as record:
        read_scene(_write(tmp_path, _effect_dae(technique)))
    assert record[0].filename == __file__


def test_read_opaque_convention_warns_once_per_document(tmp_path: Path) -> None:
    technique = (
        "<lambert><diffuse><color>0 1 0 1</color></diffuse>"
        '<transparent opaque="A_ONE"><color>1 1 1 1</color></transparent>'
        "<transparency><float>0</float></transparency></lambert>"
    )
    text = (
        _effect_dae(technique)
        .replace(
            "</library_effects>",
            f'<effect id="fx1"><profile_COMMON><technique sid="c">{technique}'
            "</technique></profile_COMMON></effect></library_effects>",
        )
        .replace(
            "</library_materials>",
            '<material id="mat1"><instance_effect url="#fx1"/></material>'
            "</library_materials>",
        )
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        scene = read_scene(_write(tmp_path, text))
    assert [str(w.message).count("convention for opaque") for w in record] == [1]
    assert [m.alpha_mode for m in scene.materials] == ["OPAQUE", "OPAQUE"]


def test_read_image_forms_1_5(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\nfake"
    images = (
        '<image id="img0"><init_from><ref>a.png</ref></init_from></image>'
        f'<image id="img1"><init_from><hex format="png">{payload.hex()}</hex>'
        "</init_from></image>"
        '<image id="img2"><init_from>data:image/jpeg;base64,'
        f"{base64.b64encode(payload).decode()}</init_from></image>"
    )
    text = _effect_dae(
        "<phong><diffuse><color>1 1 1 1</color></diffuse></phong>", images=images
    ).replace('version="1.4.1"', 'version="1.5.0"')
    scene = read_scene(_write(tmp_path, text))
    assert scene.images[0] == SceneImage(uri="a.png", media_type="image/png")
    assert scene.images[1] == SceneImage(data=payload, media_type="image/png")
    assert scene.images[2] == SceneImage(data=payload, media_type="image/jpeg")


def test_read_unbound_symbol_falls_back_to_material_id(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]], material="mat0"))
    text = _dae(
        '<library_effects><effect id="fx0"><profile_COMMON><technique sid="c">'
        "<constant/></technique></profile_COMMON></effect></library_effects>"
        '<library_materials><material id="mat0"><instance_effect url="#fx0"/>'
        "</material></library_materials>"
        f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0")
    )
    scene = read_scene(_write(tmp_path, text))
    assert scene.meshes[0].element_attrs["material"].tolist() == [0]
    assert scene.materials[0].extras["shading"] == "constant"


def test_read_unknown_symbol_is_minus_one(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]], material="nope"))
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.warns(UserWarning, match="nope"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.meshes[0].element_attrs["material"].tolist() == [-1]


def test_read_second_binding_of_a_shared_mesh_warns(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]], material="sym"))
    mats = "".join(
        f'<material id="mat{i}"><instance_effect url="#fx0"/></material>'
        for i in range(2)
    )

    def bind(i):
        return (
            "<bind_material><technique_common>"
            f'<instance_material symbol="sym" target="#mat{i}"/>'
            "</technique_common></bind_material>"
        )

    text = _dae(
        '<library_effects><effect id="fx0"><profile_COMMON><technique sid="c">'
        "<phong/></technique></profile_COMMON></effect></library_effects>"
        f"<library_materials>{mats}</library_materials>"
        f"<library_geometries>{geo}</library_geometries>"
        '<library_visual_scenes><visual_scene id="S">'
        f'<node id="a"><instance_geometry url="#g0">{bind(0)}</instance_geometry></node>'
        f'<node id="b"><instance_geometry url="#g0">{bind(1)}</instance_geometry></node>'
        "</visual_scene></library_visual_scenes>"
    )
    with pytest.warns(UserWarning, match="bound"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.meshes[0].element_attrs["material"].tolist() == [0]


def test_read_a_one_white_zero_is_the_opaque_convention(tmp_path: Path) -> None:
    technique = (
        "<lambert><diffuse><color>0 1 0 1</color></diffuse>"
        '<transparent opaque="A_ONE"><color>1 1 1 1</color></transparent>'
        "<transparency><float>0</float></transparency></lambert>"
    )
    with pytest.warns(UserWarning, match="convention for opaque"):
        mat = read_scene(_write(tmp_path, _effect_dae(technique))).materials[0]
    assert mat.base_color == (0.0, 1.0, 0.0, 1.0)
    assert mat.alpha_mode == "OPAQUE"


def test_read_1_5_sampler_instance_image(tmp_path: Path) -> None:
    newparams = (
        '<newparam sid="samp"><sampler2D><instance_image url="#img0"/>'
        "<wrap_s>CLAMP</wrap_s></sampler2D></newparam>"
    )
    technique = (
        '<phong><diffuse><texture texture="samp" texcoord="UVMap"/></diffuse></phong>'
    )
    images = '<image id="img0"><init_from><ref>a.png</ref></init_from></image>'
    text = _effect_dae(technique, newparams=newparams, images=images).replace(
        'version="1.4.1"', 'version="1.5.0"'
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scene = read_scene(_write(tmp_path, text))
    assert scene.materials[0].base_color_texture == 0
    assert scene.textures[0] == SceneTexture(image=0, wrap_s=33071)


def test_read_empty_surface_does_not_match_an_image_without_id(tmp_path: Path) -> None:
    newparams = (
        '<newparam sid="surf"><surface type="2D"><init_from></init_from>'
        "</surface></newparam>"
        '<newparam sid="samp"><sampler2D><source>surf</source></sampler2D></newparam>'
    )
    technique = (
        '<phong><diffuse><texture texture="samp" texcoord="UVMap"/></diffuse></phong>'
    )
    images = "<image><init_from>a.png</init_from></image>"
    with pytest.warns(UserWarning, match="reaches no image"):
        scene = read_scene(
            _write(tmp_path, _effect_dae(technique, newparams=newparams, images=images))
        )
    assert scene.materials[0].base_color_texture is None


def test_read_first_unbound_instance_yields_to_a_later_binding(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]], material="sym"))
    text = _dae(
        '<library_effects><effect id="fx0"><profile_COMMON><technique sid="c">'
        "<phong/></technique></profile_COMMON></effect></library_effects>"
        '<library_materials><material id="mat0"><instance_effect url="#fx0"/>'
        "</material></library_materials>"
        f"<library_geometries>{geo}</library_geometries>"
        '<library_visual_scenes><visual_scene id="S">'
        '<node id="a"><instance_geometry url="#g0"/></node>'
        '<node id="b"><instance_geometry url="#g0"><bind_material>'
        '<technique_common><instance_material symbol="sym" target="#mat0"/>'
        "</technique_common></bind_material></instance_geometry></node>"
        "</visual_scene></library_visual_scenes>"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scene = read_scene(_write(tmp_path, text))
    assert scene.meshes[0].element_attrs["material"].tolist() == [0]


# ---------------------------------------------------------------------------
# nodes, transforms, scenes
# ---------------------------------------------------------------------------


def test_read_node_transform_composition(tmp_path: Path) -> None:
    node_extra = (
        '<translate sid="location">1 2 3</translate>'
        '<rotate sid="rotationZ">0 0 1 90</rotate>'
        '<scale sid="scale">2 2 2</scale>'
    )
    text = _dae(
        "<library_geometries>"
        + _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
        + "</library_geometries>"
        + _scene_with("g0", node_extra=node_extra)
    )
    scene = read_scene(_write(tmp_path, text))
    node = scene.nodes[0]
    expected = np.array(
        [[0.0, -2, 0, 1], [2, 0, 0, 2], [0, 0, 2, 3], [0, 0, 0, 1]], dtype=np.float64
    )
    np.testing.assert_allclose(node.matrix, expected, atol=1e-12)
    assert [t["kind"] for t in node.extras["transforms"]] == [
        "translate",
        "rotate",
        "scale",
    ]
    assert node.extras["transforms"][1]["sid"] == "rotationZ"
    assert node.extras["id"] == "n0"
    flat = scene.to_polydata()
    np.testing.assert_allclose(flat.vertices[1], [1, 4, 3], atol=1e-12)


def test_read_matrix_is_row_major(tmp_path: Path) -> None:
    m = np.arange(16, dtype=np.float64).reshape(4, 4)
    m[3] = [0, 0, 0, 1]
    text = _dae(
        "<library_geometries>"
        + _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
        + "</library_geometries>"
        + _scene_with("g0", node_extra=f'<matrix sid="transform">{_nums(m)}</matrix>')
    )
    node = read_scene(_write(tmp_path, text)).nodes[0]
    np.testing.assert_array_equal(node.matrix, m)


def test_read_lookat_and_skew(tmp_path: Path) -> None:
    node_extra = "<lookat>0 0 5 0 0 0 0 1 0</lookat><skew>10 1 0 0 0 1 0</skew>"
    text = _dae(
        "<library_geometries>"
        + _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
        + "</library_geometries>"
        + _scene_with("g0", node_extra=node_extra)
    )
    with pytest.warns(UserWarning, match="skew"):
        node = read_scene(_write(tmp_path, text)).nodes[0]
    np.testing.assert_allclose(node.matrix[:3, 3], [0, 0, 5])
    np.testing.assert_allclose(node.matrix[:3, :3], np.eye(3), atol=1e-12)


def test_read_hierarchy_instance_node_and_multiple_geometries(tmp_path: Path) -> None:
    geos = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]), name="A") + _geometry(
        "g1", _QUAD, _triangles("g1", [[0, 1, 2]]), name="B"
    )
    text = _dae(
        f"<library_geometries>{geos}</library_geometries>"
        '<library_nodes><node id="lib" name="Lib"><translate>0 0 1</translate>'
        '<instance_geometry url="#g1"/></node></library_nodes>'
        '<library_visual_scenes><visual_scene id="S" name="Main">'
        '<node id="root" name="Root" sid="r" type="JOINT">'
        '<node id="kid" name="Kid"><instance_geometry url="#g0"/>'
        '<instance_geometry url="#g1"/></node>'
        '<instance_node url="#lib"/><instance_camera url="#cam"/>'
        "</node></visual_scene>"
        '<visual_scene id="S2" name="Second"><node id="other"/></visual_scene>'
        "</library_visual_scenes>"
        '<scene><instance_visual_scene url="#S2"/></scene>'
    )
    scene = read_scene(_write(tmp_path, text))
    assert scene.scenes == ((0,), (4,))
    assert scene.active_scene == 1
    assert scene.name == "Second"
    root, kid, extra, lib, other = scene.nodes
    assert root.name == "Root" and root.mesh is None
    assert root.extras["sid"] == "r" and root.extras["type"] == "JOINT"
    assert root.extras["camera"] == "cam"
    assert root.children == (1, 3)
    assert kid.mesh == 0 and kid.children == (2,)
    assert extra.name == "B" and extra.mesh == 1
    assert lib.name == "Lib" and lib.mesh == 1
    np.testing.assert_array_equal(lib.matrix[:3, 3], [0, 0, 1])
    assert other.name == "" and other.children == ()


def test_geometry_name_is_mesh_name_both_ways(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]), name="Hull")
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    scene = read_scene(_write(tmp_path, text))
    assert scene.meshes[0].global_attrs == {"mesh_name": "Hull"}
    out = tmp_path / "named.dae"
    write_scene(scene, out)
    assert '<geometry id="geometry0" name="Hull">' in out.read_text()
    assert read_scene(out).meshes[0].global_attrs["mesh_name"] == "Hull"


def test_read_every_library_of_a_kind(tmp_path: Path) -> None:
    geos = (
        f"<library_geometries>{_geometry('g0', _TRI, _triangles('g0', [[0, 1, 2]]))}"
        "</library_geometries><library_geometries>"
        f"{_geometry('g1', _QUAD, _triangles('g1', [[0, 1, 2]], material='sym'))}"
        "</library_geometries>"
    )
    effects = (
        '<library_effects><effect id="fx"><profile_COMMON><technique sid="c">'
        "<lambert><diffuse><color>1 0 0 1</color></diffuse></lambert>"
        "</technique></profile_COMMON></effect></library_effects>"
    )
    materials = (
        '<library_materials><material id="m0"><instance_effect url="#fx"/>'
        "</material></library_materials><library_materials>"
        '<material id="m1"><instance_effect url="#fx"/></material>'
        "</library_materials>"
    )
    images = (
        '<library_images><image id="i0"><init_from>a.png</init_from></image>'
        '</library_images><library_images><image id="i1"><init_from>b.png'
        "</init_from></image></library_images>"
    )
    bind = (
        '<bind_material><technique_common><instance_material symbol="sym" '
        'target="#m1"/></technique_common></bind_material>'
    )
    scenes = (
        '<library_visual_scenes><visual_scene id="S0"><node id="n0">'
        '<instance_geometry url="#g0"/></node></visual_scene></library_visual_scenes>'
        '<library_visual_scenes><visual_scene id="S1"><node id="n1">'
        '<translate sid="loc">0 0 0</translate>'
        f'<instance_geometry url="#g1">{bind}</instance_geometry></node>'
        "</visual_scene></library_visual_scenes>"
        '<scene><instance_visual_scene url="#S1"/></scene>'
    )
    anims = (
        "<library_animations>"
        + '<animation id="a0">'
        + _source("a0-t", [0.0, 1.0], "TIME")
        + _source("a0-v", np.zeros((2, 3)), "X Y Z")
        + '<sampler id="a0-s"><input semantic="INPUT" source="#a0-t"/>'
        '<input semantic="OUTPUT" source="#a0-v"/></sampler>'
        '<channel source="#a0-s" target="n1/loc"/></animation>'
        "</library_animations><library_animations>"
        '<animation id="a1">'
        + _source("a1-t", [0.0, 1.0], "TIME")
        + _source("a1-v", np.ones((2, 3)), "X Y Z")
        + '<sampler id="a1-s"><input semantic="INPUT" source="#a1-t"/>'
        '<input semantic="OUTPUT" source="#a1-v"/></sampler>'
        '<channel source="#a1-s" target="n1/loc"/></animation>'
        "</library_animations>"
    )
    text = _dae(images + effects + materials + geos + scenes + anims)
    scene = read_scene(_write(tmp_path, text))
    assert len(scene.meshes) == 2
    assert [img.uri for img in scene.images] == ["a.png", "b.png"]
    assert len(scene.materials) == 2
    assert scene.meshes[1].element_attrs["material"].tolist() == [1]
    assert scene.scenes == ((0,), (1,)) and scene.active_scene == 1
    assert [a["name"] for a in scene.global_attrs["animations"]] == ["a0", "a1"]


def test_read_instance_node_cycle_refused(tmp_path: Path) -> None:
    text = _dae(
        '<library_nodes><node id="a"><instance_node url="#b"/></node>'
        '<node id="b"><instance_node url="#a"/></node></library_nodes>'
        '<library_visual_scenes><visual_scene id="S"><instance_node url="#a"/>'
        "</visual_scene></library_visual_scenes>"
    )
    with pytest.raises(CodecError, match="itself"):
        read_scene(_write(tmp_path, text))


def test_read_no_visual_scene_synthesises_nodes(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    scene = read_scene(
        _write(tmp_path, _dae(f"<library_geometries>{geo}</library_geometries>"))
    )
    assert scene.nodes == (SceneNode(mesh=0),)
    assert scene.scenes == ((0,),)


def test_read_asset(tmp_path: Path) -> None:
    asset = (
        "<contributor><authoring_tool>Blender</authoring_tool></contributor>"
        "<created>2020-01-01T00:00:00</created>"
        '<unit name="centimeter" meter="0.01"/><up_axis>Z_UP</up_axis>'
    )
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    scene = read_scene(
        _write(
            tmp_path,
            _dae(f"<library_geometries>{geo}</library_geometries>", asset=asset),
        )
    )
    assert scene.global_attrs["asset"] == {
        "up_axis": "Z_UP",
        "unit": {"name": "centimeter", "meter": 0.01},
        "authoring_tool": "Blender",
        "created": "2020-01-01T00:00:00",
    }


def test_read_instance_node_of_a_geometry_refused(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    text = _dae(
        f"<library_geometries>{geo}</library_geometries>"
        '<library_visual_scenes><visual_scene id="S">'
        '<node id="n"><instance_node url="#g0"/></node>'
        "</visual_scene></library_visual_scenes>"
    )
    with pytest.raises(CodecError, match="not a node"):
        read_scene(_write(tmp_path, text))


# ---------------------------------------------------------------------------
# skins
# ---------------------------------------------------------------------------


def _skin_dae(vcount: str, v: str, *, weights=(0.5, 0.25, 0.25, 0.1, 0.9)) -> str:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    ibm = np.tile(np.eye(4).ravel(), (2, 1))
    ibm[1, 3] = -1.0
    ctrl = (
        '<library_controllers><controller id="ctrl" name="Armature">'
        '<skin source="#g0"><bind_shape_matrix>'
        "1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1</bind_shape_matrix>"
        + _source("ctrl-joints", ["hip", "knee"], "JOINT", kind="Name_array")
        + _source("ctrl-ibm", ibm, "TRANSFORM")
        + _source("ctrl-w", np.asarray(weights), "WEIGHT")
        + '<joints><input semantic="JOINT" source="#ctrl-joints"/>'
        '<input semantic="INV_BIND_MATRIX" source="#ctrl-ibm"/></joints>'
        f'<vertex_weights count="3"><input semantic="JOINT" source="#ctrl-joints"'
        ' offset="0"/><input semantic="WEIGHT" source="#ctrl-w" offset="1"/>'
        f"<vcount>{vcount}</vcount><v>{v}</v></vertex_weights>"
        "</skin></controller></library_controllers>"
    )
    return _dae(
        f"<library_geometries>{geo}</library_geometries>{ctrl}"
        '<library_visual_scenes><visual_scene id="S">'
        '<node id="arm" name="Armature"><node id="hipn" sid="hip" type="JOINT">'
        '<node id="kneen" sid="knee" type="JOINT"/></node></node>'
        '<node id="body"><instance_controller url="#ctrl"><skeleton>#hipn</skeleton>'
        "</instance_controller></node>"
        "</visual_scene></library_visual_scenes>"
    )


def test_read_skin(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _skin_dae("2 1 1", "0 0 1 1 1 2 -1 3")))
    body = scene.nodes[3]
    assert body.mesh == 0 and body.extras["skin"] == 0
    skin = scene.global_attrs["skins"][0]
    assert skin["name"] == "Armature"
    assert skin["joints"] == [1, 2]
    assert skin["mesh"] == 0
    np.testing.assert_array_equal(skin["bind_shape_matrix"][:3, 3], [0, 0, 5])
    assert skin["inverse_bind_matrices"].shape == (2, 4, 4)
    assert skin["inverse_bind_matrices"][1, 0, 3] == -1.0
    mesh = scene.meshes[0]
    assert mesh.vertex_attrs["joints"].dtype == np.int32
    np.testing.assert_array_equal(
        mesh.vertex_attrs["joints"], [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
    )
    np.testing.assert_allclose(
        mesh.vertex_attrs["weights"],
        [[0.5, 0.25, 0, 0], [0.25, 0, 0, 0], [0, 0, 0, 0]],
    )


def test_read_skin_more_than_four_influences_trimmed(tmp_path: Path) -> None:
    text = _skin_dae("5 0 0", "0 0 1 1 0 2 1 3 0 4")
    with pytest.warns(UserWarning, match="influences"):
        mesh = read_scene(_write(tmp_path, text)).meshes[0]
    w = mesh.vertex_attrs["weights"][0]
    assert w.sum() == pytest.approx(1.0)
    assert mesh.vertex_attrs["joints"][0].tolist() == [0, 0, 1, 0]
    np.testing.assert_allclose(w, np.array([0.9, 0.5, 0.25, 0.25]) / 1.9)


def test_read_skin_covering_fewer_vertices_warns(tmp_path: Path) -> None:
    text = _skin_dae("2 1", "0 0 1 1 1 2").replace(
        '<vertex_weights count="3">', '<vertex_weights count="2">'
    )
    with pytest.warns(UserWarning, match="covers 2 of the geometry's 3"):
        mesh = read_scene(_write(tmp_path, text)).meshes[0]
    np.testing.assert_array_equal(mesh.vertex_attrs["weights"][2], 0.0)
    np.testing.assert_allclose(mesh.vertex_attrs["weights"][0], [0.5, 0.25, 0, 0])


def test_read_skin_joints_sharing_a_name_keep_their_slots(tmp_path: Path) -> None:
    text = _skin_dae("1 1 1", "0 0 1 1 1 2").replace(
        ">hip knee</Name_array>", ">hip hip</Name_array>"
    )
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    # Both joints keep their own slot (and inverse bind matrix); the name
    # only decides which node each binds to.
    assert mesh.vertex_attrs["joints"][:, 0].tolist() == [0, 1, 1]


def test_read_skin_unknown_joint_warns(tmp_path: Path) -> None:
    text = _skin_dae("1 1 1", "0 0 1 1 1 2").replace('sid="knee"', 'sid="ankle"')
    with pytest.warns(UserWarning, match="knee"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.global_attrs["skins"][0]["joints"] == [1, -1]


def test_read_skin_v_length_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="vertex_weights"):
        read_scene(_write(tmp_path, _skin_dae("2 1 1", "0 0 1 1 1 2")))


def test_read_morph_controller_warns(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    text = _dae(
        f"<library_geometries>{geo}</library_geometries>"
        '<library_controllers><controller id="m"><morph source="#g0"/></controller>'
        "</library_controllers>"
        '<library_visual_scenes><visual_scene id="S"><node id="n">'
        '<instance_controller url="#m"/></node></visual_scene></library_visual_scenes>'
    )
    with pytest.warns(UserWarning, match="morph"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.nodes[0].mesh == 0
    assert "skins" not in scene.global_attrs


def test_read_skin_non_integer_v_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="<v> is not integers"):
        read_scene(_write(tmp_path, _skin_dae("2 1 1", "0 0 1.5 1 1 2 -1 3")))
    with pytest.raises(CodecError, match="<vcount> is not integers"):
        read_scene(_write(tmp_path, _skin_dae("2 1e0 1", "0 0 1 1 1 2 -1 3")))
    with pytest.raises(CodecError, match="<vcount> has an entry below 0"):
        read_scene(_write(tmp_path, _skin_dae("2 -1 3", "0 0 1 1 1 2 -1 3")))


def _two_rigs_dae() -> str:
    geos = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]])) + _geometry(
        "g1", _TRI, _triangles("g1", [[0, 1, 2]])
    )

    def ctrl(cid: str, gid: str) -> str:
        return (
            f'<controller id="{cid}"><skin source="#{gid}">'
            + _source(f"{cid}-joints", ["hip", "knee"], "JOINT", kind="Name_array")
            + _source(f"{cid}-w", np.array([1.0]), "WEIGHT")
            + f'<joints><input semantic="JOINT" source="#{cid}-joints"/></joints>'
            f'<vertex_weights count="3"><input semantic="JOINT" source="#{cid}-joints"'
            f' offset="0"/><input semantic="WEIGHT" source="#{cid}-w" offset="1"/>'
            "<vcount>1 1 1</vcount><v>0 0 1 0 1 0</v></vertex_weights>"
            "</skin></controller>"
        )

    def rig(prefix: str) -> str:
        return (
            f'<node id="{prefix}hip" sid="hip" type="JOINT">'
            f'<node id="{prefix}knee" sid="knee" type="JOINT"/></node>'
        )

    return _dae(
        f"<library_geometries>{geos}</library_geometries>"
        f"<library_controllers>{ctrl('c0', 'g0')}{ctrl('c1', 'g1')}</library_controllers>"
        '<library_visual_scenes><visual_scene id="S">'
        f'<node id="body0"><instance_controller url="#c0"><skeleton>#a_hip</skeleton>'
        "</instance_controller></node>"
        f'<node id="body1"><instance_controller url="#c1"><skeleton>#b_hip</skeleton>'
        "</instance_controller></node>"
        f'<node id="rigA">{rig("a_")}</node><node id="rigB">{rig("b_")}</node>'
        "</visual_scene></library_visual_scenes>"
    )


def test_read_skeleton_scopes_joint_sids_to_its_rig(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _two_rigs_dae()))
    ids = [n.extras.get("id") for n in scene.nodes]
    skins = scene.global_attrs["skins"]
    assert [ids[j] for j in skins[0]["joints"]] == ["a_hip", "a_knee"]
    assert [ids[j] for j in skins[1]["joints"]] == ["b_hip", "b_knee"]
    out = tmp_path / "rigs.dae"
    write_scene(scene, out)
    back = read_scene(out)
    back_ids = [n.extras.get("id") for n in back.nodes]
    assert [back_ids[j] for j in back.global_attrs["skins"][1]["joints"]] == [
        "b_hip",
        "b_knee",
    ]


def test_read_joint_named_by_id_binds_inside_its_skeleton(tmp_path: Path) -> None:
    text = (
        _two_rigs_dae()
        .replace("hip knee", "a_hip a_knee")
        .replace('<node id="rigB">', '<node id="rigB" sid="a_hip">')
    )
    scene = read_scene(_write(tmp_path, text))
    ids = [n.extras.get("id") for n in scene.nodes]
    assert [ids[j] for j in scene.global_attrs["skins"][0]["joints"]] == [
        "a_hip",
        "a_knee",
    ]


def test_read_second_controller_on_a_node_keeps_its_skin(tmp_path: Path) -> None:
    text = _two_rigs_dae().replace(
        '</instance_controller></node><node id="body1">', "</instance_controller>"
    )
    scene = read_scene(_write(tmp_path, text))
    body = scene.nodes[0]
    assert body.mesh == 0 and body.extras["skin"] == 0
    child = scene.nodes[body.children[0]]
    assert child.mesh == 1 and child.extras["skin"] == 1


# ---------------------------------------------------------------------------
# animations
# ---------------------------------------------------------------------------


def _anim_dae(
    channels: str, *, nested: bool = False, interp=("LINEAR", "LINEAR")
) -> str:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    times = [0.0, 1.0]
    tr = np.array([[0.0, 0, 0], [1, 2, 3]])
    ang = [0.0, 90.0]
    mats = np.tile(np.eye(4).ravel(), (2, 1))
    body = (
        _source("a-t", times, "TIME")
        + _source("a-tr", tr, "X Y Z")
        + _source("a-ang", ang, "ANGLE")
        + _source("a-m", mats, "TRANSFORM")
        + _source("a-i", list(interp), "INTERPOLATION", kind="Name_array")
        + '<sampler id="s-tr"><input semantic="INPUT" source="#a-t"/>'
        '<input semantic="OUTPUT" source="#a-tr"/>'
        '<input semantic="INTERPOLATION" source="#a-i"/></sampler>'
        '<sampler id="s-ang"><input semantic="INPUT" source="#a-t"/>'
        '<input semantic="OUTPUT" source="#a-ang"/></sampler>'
        '<sampler id="s-m"><input semantic="INPUT" source="#a-t"/>'
        '<input semantic="OUTPUT" source="#a-m"/></sampler>' + channels
    )
    anim = (
        f'<animation id="anim" name="Walk"><animation id="inner">{body}</animation>'
        "</animation>"
        if nested
        else f'<animation id="anim" name="Walk">{body}</animation>'
    )
    return _dae(
        f"<library_geometries>{geo}</library_geometries>"
        f"<library_animations>{anim}</library_animations>"
        + _scene_with(
            "g0",
            node_extra='<translate sid="location">0 0 0</translate>'
            '<rotate sid="rotationZ">0 0 1 0</rotate>'
            '<node id="n1" sid="child"><matrix sid="transform">'
            "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix></node>",
        )
    )


def test_read_animation_channels(tmp_path: Path) -> None:
    channels = (
        '<channel source="#s-tr" target="n0/location"/>'
        '<channel source="#s-ang" target="n0/rotationZ.ANGLE"/>'
        '<channel source="#s-m" target="n0/child/transform"/>'
    )
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    anims = scene.global_attrs["animations"]
    assert len(anims) == 1
    anim = anims[0]
    assert anim["name"] == "Walk"
    assert [c["target"] for c in anim["channels"]] == [
        {"node": 0, "path": "translation", "sid": "location", "member": None},
        {"node": 0, "path": "rotation", "sid": "rotationZ", "member": "ANGLE"},
        {"node": 1, "path": "matrix", "sid": "transform", "member": None},
    ]
    assert [c["sampler"] for c in anim["channels"]] == [0, 1, 2]
    s = anim["samplers"]
    np.testing.assert_array_equal(s[0]["times"], [0.0, 1.0])
    np.testing.assert_array_equal(s[0]["values"], [[0, 0, 0], [1, 2, 3]])
    assert s[0]["interpolation"] == "LINEAR"
    np.testing.assert_array_equal(s[1]["values"], [0.0, 90.0])
    assert s[1]["interpolation"] == "LINEAR"
    assert s[2]["values"].shape == (2, 16)


def test_read_animation_nested_and_bezier_tangents(tmp_path: Path) -> None:
    channels = '<channel source="#s-tr" target="n0/location"/>'
    text = _anim_dae(channels, nested=True, interp=("BEZIER", "BEZIER")).replace(
        '<input semantic="INTERPOLATION" source="#a-i"/>',
        '<input semantic="INTERPOLATION" source="#a-i"/>'
        '<input semantic="IN_TANGENT" source="#a-tr"/>'
        '<input semantic="OUT_TANGENT" source="#a-tr"/>',
    )
    anim = read_scene(_write(tmp_path, text)).global_attrs["animations"][0]
    assert anim["name"] == "Walk"
    s = anim["samplers"][0]
    assert s["interpolation"] == "BEZIER"
    assert s["in_tangents"].shape == (2, 3)
    assert s["out_tangents"].shape == (2, 3)


def test_read_animation_unresolved_target_dropped(tmp_path: Path) -> None:
    channels = (
        '<channel source="#s-tr" target="nowhere/location"/>'
        '<channel source="#s-tr" target="n0/nosuchsid"/>'
        '<channel source="#s-ang" target="n0/rotationZ.ANGLE"/>'
    )
    with pytest.warns(UserWarning) as rec:
        anim = read_scene(_write(tmp_path, _anim_dae(channels))).global_attrs[
            "animations"
        ][0]
    assert any("nowhere" in str(w.message) for w in rec)
    assert any("nosuchsid" in str(w.message) for w in rec)
    assert len(anim["channels"]) == 1
    assert anim["channels"][0]["sampler"] == 0
    assert len(anim["samplers"]) == 1


def test_read_animation_times_values_mismatch_refused(tmp_path: Path) -> None:
    channels = '<channel source="#s-tr" target="n0/location"/>'
    text = (
        _anim_dae(channels)
        .replace(
            '<float_array id="a-t-array" count="2">0.0 1.0',
            '<float_array id="a-t-array" count="3">0.0 1.0 2.0',
        )
        .replace(
            '<accessor source="#a-t-array" count="2"',
            '<accessor source="#a-t-array" count="3"',
        )
    )
    with pytest.raises(CodecError, match="keys"):
        read_scene(_write(tmp_path, text))


def test_no_animation_key_when_absent(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _tri_dae()))
    assert "animations" not in scene.global_attrs


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_refuse_not_xml(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="well-formed"):
        read_scene(_write(tmp_path, "<COLLADA"))


def test_refuse_wrong_root(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="<COLLADA>"):
        read_scene(_write(tmp_path, '<?xml version="1.0"?><model/>'))


def test_refuse_unknown_version(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="version"):
        read_scene(
            _write(tmp_path, _tri_dae().replace('version="1.4.1"', 'version="2.0"'))
        )


def test_refuse_float_array_count_mismatch(tmp_path: Path) -> None:
    text = _tri_dae().replace('count="9"', 'count="12"')
    with pytest.raises(CodecError, match="12"):
        read_scene(_write(tmp_path, text))


def test_refuse_accessor_past_array(tmp_path: Path) -> None:
    text = _tri_dae().replace('count="3" stride="3"', 'count="4" stride="3"')
    with pytest.raises(CodecError, match="accessor"):
        read_scene(_write(tmp_path, text))


def test_refuse_index_past_source(tmp_path: Path) -> None:
    text = _tri_dae().replace("<p>0 1 2</p>", "<p>0 1 3</p>")
    with pytest.raises(CodecError, match="3"):
        read_scene(_write(tmp_path, text))


def test_refuse_p_length_mismatch(tmp_path: Path) -> None:
    text = _tri_dae().replace("<p>0 1 2</p>", "<p>0 1 2 0</p>")
    with pytest.raises(CodecError, match="triangles"):
        read_scene(_write(tmp_path, text))


def test_refuse_vcount_mismatch(tmp_path: Path) -> None:
    prim = (
        '<polylist count="2"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<vcount>3</vcount><p>0 1 2</p></polylist>"
    )
    geo = _geometry("g0", _TRI, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.raises(CodecError, match="vcount"):
        read_scene(_write(tmp_path, text))


def test_refuse_dangling_url(tmp_path: Path) -> None:
    text = _tri_dae().replace('url="#g0"', 'url="#g9"')
    with pytest.raises(CodecError, match="g9"):
        read_scene(_write(tmp_path, text))


def test_refuse_bad_matrix(tmp_path: Path) -> None:
    text = _dae(
        "<library_geometries>"
        + _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
        + "</library_geometries>"
        + _scene_with("g0", node_extra="<matrix>1 2 3</matrix>")
    )
    with pytest.raises(CodecError, match="16"):
        read_scene(_write(tmp_path, text))


def test_refuse_not_numbers(tmp_path: Path) -> None:
    text = _tri_dae().replace("0.0 0.0 0.0 1.0", "0.0 x 0.0 1.0")
    with pytest.raises(CodecError, match="not numbers"):
        read_scene(_write(tmp_path, text))


def test_read_unit_with_decimal_comma_warns(tmp_path: Path) -> None:
    text = _tri_dae().replace(
        "<asset></asset>", '<asset><unit name="centimeter" meter="0,01"/></asset>'
    )
    with pytest.warns(UserWarning, match="decimal comma"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.global_attrs["asset"]["unit"] == {"name": "centimeter", "meter": 0.01}


def test_write_mixed_surface_keeps_element_order(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], float)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 4]])),
            ("quad", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[1, 2, 4]])),
        ],
    )
    out = tmp_path / "mixed.dae"
    write(poly, out)
    back = read_scene(out).meshes[0]
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    assert out.read_text().count("<polylist") == 1
    assert "<triangles" not in out.read_text()


def test_refuse_unknown_up_axis(tmp_path: Path) -> None:
    text = _tri_dae().replace(
        "<asset></asset>", "<asset><up_axis>W_UP</up_axis></asset>"
    )
    with pytest.raises(CodecError, match="W_UP"):
        read_scene(_write(tmp_path, text))


def test_lazy_refused(tmp_path: Path) -> None:
    with pytest.raises(LazyReadError):
        read(_write(tmp_path, _tri_dae()), lazy=True)


def test_read_flattens_with_warning(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="read_scene"):
        poly = read(_write(tmp_path, _tri_dae()))
    np.testing.assert_array_equal(poly.vertices, _TRI)


def test_read_from_buffer(tmp_path: Path) -> None:
    scene = read_scene(io.BytesIO(_tri_dae().encode()))
    assert len(scene.meshes) == 1


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


def _material_scene() -> SceneData:
    mesh = _surface()
    mesh = make_polydata(
        mesh.vertices,
        [("triangle", np.array([[0, 1, 4]])), ("quad", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={
            "normals": np.tile([[0.0, 0, 1]], (5, 1)),
            "texcoords": np.zeros((5, 2)),
            "colors": np.full((5, 3), 0.5),
        },
        element_attrs={"material": np.array([0, 1], dtype=np.int32)},
    )
    child = SceneNode(name="Child", mesh=0, matrix=np.diag([2.0, 2.0, 2.0, 1.0]))
    root_m = np.eye(4)
    root_m[:3, 3] = [1, 2, 3]
    root = SceneNode(name="Root", children=(1,), matrix=root_m)
    return SceneData(
        meshes=(mesh,),
        nodes=(root, child),
        materials=(
            SceneMaterial(
                name="Red",
                base_color=(1.0, 0.0, 0.0, 0.5),
                emissive=(0.1, 0.1, 0.1),
                alpha_mode="BLEND",
                double_sided=True,
                base_color_texture=0,
            ),
            SceneMaterial(
                name="Green", base_color=(0.0, 1.0, 0.0, 1.0), normal_texture=1
            ),
        ),
        textures=(SceneTexture(image=0, wrap_s=33071), SceneTexture(image=1)),
        images=(
            SceneImage(uri="tex.png", media_type="image/png", name="tex"),
            SceneImage(data=b"\x89PNGxx", media_type="image/png"),
        ),
        scenes=((0,),),
        name="Main",
        global_attrs={
            "asset": {"up_axis": "Z_UP", "unit": {"name": "cm", "meter": 0.01}}
        },
    )


def _with_animations(scene: SceneData, anims: list) -> SceneData:
    return SceneData(
        meshes=scene.meshes,
        nodes=scene.nodes,
        materials=scene.materials,
        textures=scene.textures,
        images=scene.images,
        scenes=scene.scenes,
        name=scene.name,
        global_attrs={"animations": anims},
    )


def test_write_scene_roundtrip(tmp_path: Path) -> None:
    scene = _material_scene()
    out = tmp_path / "out.dae"
    write_scene(scene, out)
    back = read_scene(out)
    assert back.name == "Main"
    assert back.scenes == ((0,),)
    assert len(back.nodes) == 2
    np.testing.assert_array_equal(back.nodes[0].matrix, scene.nodes[0].matrix)
    np.testing.assert_array_equal(back.nodes[1].matrix, scene.nodes[1].matrix)
    assert back.nodes[0].children == (1,)
    assert back.nodes[1].mesh == 0
    mesh = back.meshes[0]
    np.testing.assert_array_equal(mesh.vertices, scene.meshes[0].vertices)
    np.testing.assert_array_equal(mesh.connectivity, scene.meshes[0].connectivity)
    np.testing.assert_array_equal(mesh.element_types, scene.meshes[0].element_types)
    for key in ("normals", "texcoords", "colors"):
        np.testing.assert_array_equal(
            mesh.vertex_attrs[key], scene.meshes[0].vertex_attrs[key]
        )
    assert mesh.element_attrs["material"].tolist() == [0, 1]
    red, green = back.materials
    assert red.name == "Red" and red.base_color == (1.0, 1.0, 1.0, 0.5)
    assert red.alpha_mode == "BLEND" and red.double_sided
    assert red.emissive == pytest.approx((0.1, 0.1, 0.1))
    assert red.base_color_texture == 0 and green.normal_texture == 1
    assert back.textures[0].wrap_s == 33071
    assert back.images[0] == scene.images[0]
    assert back.images[1] == scene.images[1]
    assert back.global_attrs["asset"]["up_axis"] == "Z_UP"
    assert back.global_attrs["asset"]["unit"] == {"name": "cm", "meter": 0.01}


def test_write_is_deterministic_and_buffer_equal(tmp_path: Path) -> None:
    scene = _material_scene()
    out = tmp_path / "a.dae"
    write_scene(scene, out)
    buf = io.BytesIO()
    write_scene(scene, buf)
    assert out.read_bytes() == buf.getvalue()
    write_scene(scene, tmp_path / "b.dae")
    assert out.read_bytes() == (tmp_path / "b.dae").read_bytes()


def test_write_flat_polydata(tmp_path: Path) -> None:
    poly = make_polydata(
        _surface().vertices,
        [("triangle", np.array([[0, 1, 4]])), ("quad", np.array([[0, 1, 2, 3]]))],
        element_attrs={"material": np.array([2, -1], dtype=np.int32)},
    )
    out = tmp_path / "flat.dae"
    write(poly, out)
    back = read_scene(out)
    assert len(back.meshes) == 1 and len(back.nodes) == 1
    assert back.meshes[0].element_attrs["material"].tolist() == [0, -1]
    assert back.materials[0].name == "material_2"
    with pytest.warns(UserWarning, match="read_scene"):
        flat = read(out)
    np.testing.assert_array_equal(flat.vertices, poly.vertices)


def test_write_lines_strips_and_points(tmp_path: Path) -> None:
    poly = make_polydata(
        _QUAD,
        [
            ("line", np.array([[0, 1]])),
            ("poly_line", np.array([[0, 1, 2, 3]])),
            ("triangle_strip", np.array([[0, 1, 3, 2]])),
            ("vertex", np.array([[0]])),
        ],
    )
    out = tmp_path / "l.dae"
    with pytest.warns(UserWarning, match="vertex"):
        write(poly, out)
    back = read_scene(out).meshes[0]
    assert back.element_types.tolist() == [
        ELEMENT_TYPES["line"],
        ELEMENT_TYPES["poly_line"],
        ELEMENT_TYPES["triangle_strip"],
    ]
    assert back.connectivity.tolist() == [0, 1, 0, 1, 2, 3, 0, 1, 3, 2]


def test_write_volume_elements_skipped(tmp_path: Path) -> None:
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 2]]))],
    )
    out = tmp_path / "v.dae"
    with pytest.warns(UserWarning, match="tetra"):
        write(poly, out)
    assert len(read_scene(out).meshes[0].element_types) == 1


def test_write_nothing_writable_keeps_the_positions(tmp_path: Path) -> None:
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for kind, cells in (("tetra", [[0, 1, 2, 3]]), ("vertex", [[0], [1], [2], [3]])):
        poly = make_polydata(verts, [(kind, np.array(cells))])
        out = tmp_path / f"{kind}.dae"
        with pytest.warns(UserWarning, match=rf"\['{kind}'\].*leaving the positions"):
            write(poly, out)
        back = read_scene(out).meshes[0]
        np.testing.assert_array_equal(back.vertices, verts)
        assert len(back.element_types) == 0


def test_write_other_vertex_attrs_dropped_with_warning(tmp_path: Path) -> None:
    poly = make_polydata(
        _TRI, [("triangle", np.array([[0, 1, 2]]))], vertex_attrs={"temp": np.ones(3)}
    )
    with pytest.warns(UserWarning, match="temp"):
        write(poly, tmp_path / "t.dae")


def test_write_texcoord_set_that_is_not_a_number_gets_a_free_one(
    tmp_path: Path,
) -> None:
    uv = np.zeros((3, 2))
    poly = make_polydata(
        _TRI,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={
            "texcoords_uv2": uv + 2,
            "texcoords_1": uv + 1,
            "texcoords": uv,
            "texcoords_0": uv + 3,
        },
    )
    out = tmp_path / "sets.dae"
    with pytest.warns(
        UserWarning, match=r"texcoords_uv2 as set 2.*texcoords_0 as set 3"
    ):
        write(poly, out)
    inputs = re.findall(
        r'semantic="TEXCOORD" source="#([^"]*)"[^>]*set="([^"]*)"', out.read_text()
    )
    assert inputs == [
        ("geometry0-texcoords_2", "2"),
        ("geometry0-texcoords_1", "1"),
        ("geometry0-texcoords", "0"),
        ("geometry0-texcoords_3", "3"),
    ]
    back = read_scene(out).meshes[0].vertex_attrs
    assert sorted(back) == ["texcoords", "texcoords_1", "texcoords_2", "texcoords_3"]
    np.testing.assert_array_equal(back["texcoords_2"], uv + 2)
    np.testing.assert_array_equal(back["texcoords_3"], uv + 3)


def test_write_texture_shared_by_two_slots_declares_one_sampler(tmp_path: Path) -> None:
    scene = _material_scene()
    shared = SceneMaterial(name="Glow", base_color_texture=0, emissive_texture=0)
    scene = SceneData(
        meshes=scene.meshes,
        nodes=scene.nodes,
        materials=(shared, scene.materials[1]),
        textures=scene.textures,
        images=scene.images,
        scenes=scene.scenes,
    )
    out = tmp_path / "shared.dae"
    write_scene(scene, out)
    text = out.read_text()
    assert text.count('<newparam sid="effect0-surface0">') == 1
    assert text.count('<newparam sid="effect0-sampler0">') == 1
    back = read_scene(out).materials[0]
    assert back.base_color_texture == 0 and back.emissive_texture == 0


def test_write_fully_transparent_material_roundtrips(tmp_path: Path) -> None:
    scene = _material_scene()
    clear = SceneMaterial(name="Clear", base_color=(1.0, 0.0, 0.0, 0.0))
    scene = SceneData(
        meshes=scene.meshes,
        nodes=scene.nodes,
        materials=(clear, SceneMaterial(name="Green")),
        scenes=scene.scenes,
    )
    out = tmp_path / "clear.dae"
    write_scene(scene, out)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read_scene(out).materials[0]
    assert back.base_color == (1.0, 0.0, 0.0, 0.0)
    assert back.alpha_mode == "BLEND"


def test_write_non_finite_values_in_xml_lexical_form(tmp_path: Path) -> None:
    poly = make_polydata(
        _TRI,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={
            "normals": np.array([[np.nan, 0, 1], [np.inf, 0, 1], [-np.inf, 0, 1]])
        },
    )
    out = tmp_path / "n.dae"
    write(poly, out)
    assert ">NaN 0.0 1.0 INF 0.0 1.0 -INF 0.0 1.0<" in out.read_text()
    back = read_scene(out).meshes[0].vertex_attrs["normals"]
    assert np.isnan(back[0, 0]) and back[1, 0] == np.inf and back[2, 0] == -np.inf


def test_write_groups_elements_by_block_and_material(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], float)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 4]])),
            ("quad", np.array([[0, 1, 2, 3]])),
            ("line", np.array([[3, 4]])),
            ("triangle", np.array([[1, 2, 4]])),
            ("polygon", np.array([[0, 1, 2, 3, 4]])),
        ],
        element_attrs={"material": np.array([0, 1, 0, 1, 0], dtype=np.int32)},
    )
    out = tmp_path / "grouped.dae"
    write(poly, out)
    back = read_scene(out).meshes[0]
    # Blocks appear in first-occurrence order of (block, material); within one
    # block the elements keep their order.
    assert back.element_attrs["material"].tolist() == [0, 0, 1, 1, 0]
    assert back.offsets.tolist() == [0, 3, 8, 12, 15, 17]
    assert back.connectivity.tolist() == [
        *[0, 1, 4],
        *[0, 1, 2, 3, 4],
        *[0, 1, 2, 3],
        *[1, 2, 4],
        *[3, 4],
    ]


def test_gather_runs() -> None:
    offs = np.array([0, 3, 7, 7, 9])
    got = _collada._gather(offs[[1, 3, 0]], offs[[2, 4, 1]] - offs[[1, 3, 0]])
    assert got.tolist() == [3, 4, 5, 6, 7, 8, 0, 1, 2]
    assert _collada._gather(np.zeros(0, int), np.zeros(0, int)).tolist() == []


def test_write_material_column_regrouped_by_material(tmp_path: Path) -> None:
    poly = make_polydata(
        _QUAD,
        [("triangle", np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3], [0, 1, 3]]))],
        element_attrs={"material": np.array([0, 1, 0, 1], dtype=np.int32)},
    )
    out = tmp_path / "alt.dae"
    write(poly, out)
    back = read_scene(out).meshes[0]
    assert back.element_attrs["material"].tolist() == [0, 0, 1, 1]
    assert back.connectivity.tolist() == [0, 1, 2, 1, 2, 3, 0, 2, 3, 0, 1, 3]


def test_write_skin_roundtrip(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _skin_dae("2 1 1", "0 0 1 1 1 2 -1 3")))
    out = tmp_path / "skin.dae"
    write_scene(scene, out)
    back = read_scene(out)
    skin = back.global_attrs["skins"][0]
    assert skin["joints"] == [1, 2]
    np.testing.assert_array_equal(
        skin["inverse_bind_matrices"],
        scene.global_attrs["skins"][0]["inverse_bind_matrices"],
    )
    np.testing.assert_array_equal(
        skin["bind_shape_matrix"], scene.global_attrs["skins"][0]["bind_shape_matrix"]
    )
    np.testing.assert_array_equal(
        back.meshes[0].vertex_attrs["joints"], scene.meshes[0].vertex_attrs["joints"]
    )
    np.testing.assert_allclose(
        back.meshes[0].vertex_attrs["weights"], scene.meshes[0].vertex_attrs["weights"]
    )
    assert back.nodes[3].extras["skin"] == 0
    assert back.nodes[1].extras["type"] == "JOINT"
    assert back.nodes[1].extras["sid"] == "hip"


def test_write_joints_sharing_a_sid_each_bind_their_own_node(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _skin_dae("2 1 1", "0 0 1 1 1 2 -1 3")))
    nodes = list(scene.nodes)
    nodes[2] = dataclasses.replace(nodes[2], extras={**nodes[2].extras, "sid": "hip"})
    scene = dataclasses.replace(scene, nodes=tuple(nodes))
    out = tmp_path / "twins.dae"
    write_scene(scene, out)
    names = ET.fromstring(out.read_bytes()).iter(f"{{{_NS}}}Name_array")
    assert next(names).text.split() == ["hip", "kneen"]
    back = read_scene(out)
    assert back.global_attrs["skins"][0]["joints"] == [1, 2]
    np.testing.assert_array_equal(
        back.meshes[0].vertex_attrs["joints"], scene.meshes[0].vertex_attrs["joints"]
    )


def test_write_two_skins_instance_their_own_controllers(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _skin_dae("2 1 1", "0 0 1 1 1 2 -1 3")))
    skin = scene.global_attrs["skins"][0]
    other = {**skin, "mesh": 1, "name": "Other"}
    body2 = SceneNode(name="body2", mesh=1, extras={"skin": 1})
    scene = dataclasses.replace(
        scene,
        meshes=(*scene.meshes, scene.meshes[0]),
        nodes=(*scene.nodes, body2),
        scenes=((*scene.scenes[0], len(scene.nodes)),),
        global_attrs={"skins": [skin, other]},
    )
    out = tmp_path / "two.dae"
    write_scene(scene, out)
    back = read_scene(out)
    assert [s["name"] for s in back.global_attrs["skins"]] == ["Armature", "Other"]
    assert [s["mesh"] for s in back.global_attrs["skins"]] == [0, 1]
    assert back.nodes[3].extras["skin"] == 0
    assert back.nodes[4].extras["skin"] == 1


def test_write_skin_inverse_bind_matrices_of_wrong_length_refused(
    tmp_path: Path,
) -> None:
    scene = read_scene(_write(tmp_path, _skin_dae("2 1 1", "0 0 1 1 1 2 -1 3")))
    skin = dict(scene.global_attrs["skins"][0])
    skin["inverse_bind_matrices"] = np.tile(np.eye(4), (3, 1, 1))
    scene = SceneData(
        meshes=scene.meshes,
        nodes=scene.nodes,
        scenes=scene.scenes,
        global_attrs={"skins": [skin]},
    )
    with pytest.raises(CodecError, match="3 inverse bind matrices for 2 joints"):
        write_scene(scene, tmp_path / "bad.dae")


def test_write_nodes_outside_every_scene_are_not_written(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _skin_dae("2 1 1", "0 0 1 1 1 2 -1 3")))
    anims = [
        {
            "name": "A",
            "channels": [{"sampler": 0, "target": {"node": 1, "path": "matrix"}}],
            "samplers": [
                {"times": np.array([0.0]), "values": np.eye(4).ravel()[None, :]}
            ],
        }
    ]
    orphaned = SceneData(
        meshes=scene.meshes,
        nodes=scene.nodes,
        scenes=((3,),),
        global_attrs={**scene.global_attrs, "animations": anims},
    )
    out = tmp_path / "orphan.dae"
    with pytest.warns(UserWarning) as record:
        write_scene(orphaned, out)
    messages = " | ".join(str(w.message) for w in record)
    assert "3 node(s) are in no scene" in messages
    assert "joint node(s) that no scene reaches" in messages
    assert "the written scene lacks" in messages
    text = out.read_text()
    assert "hipn" not in text and "<skeleton>" not in text
    with pytest.warns(UserWarning, match="missing0"):
        back = read_scene(out)
    assert len(back.nodes) == 1
    assert back.global_attrs["skins"][0]["joints"] == [-1, -1]
    assert "animations" not in back.global_attrs


def test_write_duplicate_node_ids_keep_the_first(tmp_path: Path) -> None:
    nodes = tuple(
        SceneNode(name=f"n{i}", mesh=0, extras={"id": "dup"}) for i in range(2)
    ) + (SceneNode(name="x", mesh=0, extras={"id": "x"}),)
    scene = SceneData(meshes=(_surface(),), nodes=nodes, scenes=((0, 1, 2),))
    out = tmp_path / "dup.dae"
    write_scene(scene, out)
    back = read_scene(out)
    assert [n.extras["id"] for n in back.nodes] == ["dup", "node1", "x"]


def test_write_node_id_with_a_trailing_newline_is_not_an_xml_name(
    tmp_path: Path,
) -> None:
    node = SceneNode(mesh=0, extras={"id": "foo\n"})
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    out = tmp_path / "nl.dae"
    write_scene(scene, out)
    assert 'id="foo' not in out.read_text(encoding="utf-8")
    assert read_scene(out).nodes[0].extras["id"] == "node0"


def test_write_node_id_starting_with_a_non_ascii_letter_is_kept(tmp_path: Path) -> None:
    node = SceneNode(mesh=0, extras={"id": "élan", "sid": "ré"})
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    out = tmp_path / "unicode.dae"
    write_scene(scene, out)
    back = read_scene(out).nodes[0]
    assert back.extras["id"] == "élan"
    assert back.extras["sid"] == "ré"


def test_write_names_with_line_breaks_round_trip(tmp_path: Path) -> None:
    mesh = dataclasses.replace(_surface(), global_attrs={"mesh_name": "a\tb"})
    node = SceneNode(name="line\none", mesh=0)
    scene = SceneData(meshes=(mesh,), nodes=(node,), scenes=((0,),), name="x\r\ny")
    out = tmp_path / "names.dae"
    write_scene(scene, out)
    back = read_scene(out)
    assert back.nodes[0].name == "line\none"
    assert back.meshes[0].global_attrs["mesh_name"] == "a\tb"
    assert back.name == "x\r\ny"


def test_write_node_id_shaped_like_a_generated_sub_id_keeps_ids_unique(
    tmp_path: Path,
) -> None:
    node = SceneNode(mesh=0, extras={"id": "geometry0-positions"})
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    out = tmp_path / "ids.dae"
    write_scene(scene, out)
    ids = [e.get("id") for e in ET.fromstring(out.read_bytes()).iter() if e.get("id")]
    assert len(ids) == len(set(ids))
    back = read_scene(out)
    assert back.nodes[0].extras["id"] == "geometry0-positions"
    assert len(back.meshes[0].vertices) == 5


def test_write_transform_sid_that_is_not_a_name_dropped(tmp_path: Path) -> None:
    m = np.eye(4)
    m[0, 3] = 2.0
    node = SceneNode(
        mesh=0,
        matrix=m,
        extras={
            "transforms": [
                {"kind": "translate", "sid": "my loc", "values": [2.0, 0.0, 0.0]}
            ]
        },
    )
    anims = [
        {
            "name": "A",
            "channels": [
                {
                    "sampler": 0,
                    "target": {"node": 0, "path": "translation", "sid": "my loc"},
                }
            ],
            "samplers": [{"times": np.array([0.0]), "values": np.zeros((1, 3))}],
        }
    ]
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(node,),
        scenes=((0,),),
        global_attrs={"animations": anims},
    )
    out = tmp_path / "sid.dae"
    with pytest.warns(UserWarning) as record:
        write_scene(scene, out)
    messages = " | ".join(str(w.message) for w in record)
    assert "not an XML name" in messages and "my loc" in messages
    assert "<translate>2.0 0.0 0.0</translate>" in out.read_text()
    back = read_scene(out)
    np.testing.assert_array_equal(back.nodes[0].matrix, m)
    assert "animations" not in back.global_attrs


def test_write_edited_node_does_not_warn_about_an_unused_sid(tmp_path: Path) -> None:
    m = np.eye(4)
    m[0, 3] = 5.0
    node = SceneNode(
        mesh=0,
        matrix=m,
        extras={
            "transforms": [
                {"kind": "translate", "sid": "my loc", "values": [2.0, 0.0, 0.0]}
            ]
        },
    )
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    out = tmp_path / "edited.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    assert '<matrix sid="transform">' in out.read_text()


def test_write_animation_roundtrip(tmp_path: Path) -> None:
    channels = (
        '<channel source="#s-tr" target="n0/location"/>'
        '<channel source="#s-ang" target="n0/rotationZ.ANGLE"/>'
        '<channel source="#s-m" target="n0/child/transform"/>'
    )
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    out = tmp_path / "anim.dae"
    write_scene(scene, out)
    back = read_scene(out)
    a0 = scene.global_attrs["animations"][0]
    a1 = back.global_attrs["animations"][0]
    assert a1["name"] == "Walk"
    assert [c["target"] for c in a1["channels"]] == [
        c["target"] for c in a0["channels"]
    ]
    for s0, s1 in zip(a0["samplers"], a1["samplers"], strict=True):
        np.testing.assert_array_equal(s0["times"], s1["times"])
        np.testing.assert_array_equal(s0["values"], s1["values"])
        assert s0["interpolation"] == s1["interpolation"]
    # The node's own transform elements come back as they were spelled.
    assert [t["sid"] for t in back.nodes[0].extras["transforms"]] == [
        "location",
        "rotationZ",
    ]


def test_write_sampler_shared_by_two_channels_written_once(tmp_path: Path) -> None:
    channels = (
        '<channel source="#s-tr" target="n0/location"/>'
        '<channel source="#s-tr" target="n0/child/transform"/>'
    )
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    assert [c["sampler"] for c in scene.global_attrs["animations"][0]["channels"]] == [
        0,
        0,
    ]
    out = tmp_path / "shared.dae"
    write_scene(scene, out)
    assert out.read_text().count("<sampler ") == 1
    back = read_scene(out).global_attrs["animations"][0]
    assert len(back["samplers"]) == 1
    assert [c["sampler"] for c in back["channels"]] == [0, 0]


@pytest.mark.parametrize("interp", ["LIN EAR", "<b>", "linear"])
def test_write_interpolation_outside_the_specification_refused(
    tmp_path: Path, interp: str
) -> None:
    channels = '<channel source="#s-tr" target="n0/location"/>'
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    scene.global_attrs["animations"][0]["samplers"][0]["interpolation"] = interp
    with pytest.raises(CodecError, match=re.escape(repr(interp))):
        write_scene(scene, tmp_path / "bad.dae")


def test_write_animation_channel_without_target_sid_dropped(tmp_path: Path) -> None:
    scene = _material_scene()
    anims = [
        {
            "name": "A",
            "channels": [
                {
                    "sampler": 0,
                    "target": {"node": 0, "path": "translation", "sid": "location"},
                }
            ],
            "samplers": [
                {
                    "times": np.array([0.0, 1.0]),
                    "values": np.zeros((2, 3)),
                    "interpolation": "LINEAR",
                }
            ],
        }
    ]
    scene = _with_animations(scene, anims)
    with pytest.warns(UserWarning, match="location"):
        write_scene(scene, tmp_path / "a.dae")


def test_write_matrix_path_targets_generated_transform(tmp_path: Path) -> None:
    scene = _material_scene()
    anims = [
        {
            "name": "A",
            "channels": [{"sampler": 0, "target": {"node": 1, "path": "matrix"}}],
            "samplers": [
                {
                    "times": np.array([0.0, 1.0]),
                    "values": np.tile(np.eye(4).ravel(), (2, 1)),
                    "interpolation": "STEP",
                }
            ],
        }
    ]
    scene = _with_animations(scene, anims)
    out = tmp_path / "m.dae"
    write_scene(scene, out)
    back = read_scene(out).global_attrs["animations"][0]
    assert back["channels"][0]["target"] == {
        "node": 1,
        "path": "matrix",
        "sid": "transform",
        "member": None,
    }
    assert back["samplers"][0]["interpolation"] == "STEP"


def test_write_edited_matrix_wins_over_spelled_transforms(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _anim_dae("")))
    node = scene.nodes[0]
    m = np.eye(4)
    m[0, 3] = 7.0
    edited = SceneNode(
        name=node.name,
        mesh=node.mesh,
        children=node.children,
        matrix=m,
        extras=node.extras,
    )
    scene = SceneData(
        meshes=scene.meshes, nodes=(edited, scene.nodes[1]), scenes=scene.scenes
    )
    out = tmp_path / "e.dae"
    write_scene(scene, out)
    back = read_scene(out)
    np.testing.assert_array_equal(back.nodes[0].matrix, m)


def test_write_material_index_past_materials_refused(tmp_path: Path) -> None:
    scene = _material_scene()
    scene = SceneData(meshes=scene.meshes, nodes=scene.nodes, scenes=scene.scenes)
    with pytest.raises(CodecError, match="material"):
        write_scene(scene, tmp_path / "m.dae")


def test_write_node_cycle_refused(tmp_path: Path) -> None:
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(SceneNode(children=(1,)), SceneNode(children=(0,), mesh=0)),
        scenes=((0,),),
    )
    with pytest.raises(CodecError, match="cycle"):
        write_scene(scene, tmp_path / "c.dae")


def test_write_bad_name_refused(tmp_path: Path) -> None:
    scene = SceneData(
        meshes=(_surface(),), nodes=(SceneNode(mesh=0, name="a\x00b"),), scenes=((0,),)
    )
    with pytest.raises(CodecError, match="name"):
        write_scene(scene, tmp_path / "n.dae")


def test_write_up_axis_option(tmp_path: Path) -> None:
    out = tmp_path / "z.dae"
    write(_surface(), out, up_axis="Z_UP")
    assert read_scene(out).global_attrs["asset"]["up_axis"] == "Z_UP"
    with pytest.raises(CodecError, match="up_axis"):
        write(_surface(), out, up_axis="Q_UP")


def test_write_default_asset(tmp_path: Path) -> None:
    out = tmp_path / "d.dae"
    write(_surface(), out)
    asset = read_scene(out).global_attrs["asset"]
    assert asset["up_axis"] == "Y_UP"
    assert asset["unit"] == {"name": "meter", "meter": 1.0}
    assert asset["authoring_tool"].startswith("polyxios")


def test_registry_and_public_api(tmp_path: Path) -> None:
    assert ".dae" in polyxios.supported_extensions()
    out = tmp_path / "api.dae"
    polyxios.write_scene(_material_scene(), out)
    scene = polyxios.read_scene(out)
    assert len(scene.materials) == 2
    with pytest.warns(UserWarning, match="read_scene"):
        polyxios.read(out)


def test_xml_forbidden_noncharacters() -> None:
    assert _collada._XML_FORBIDDEN.search("a\ufffeb")
    assert _collada._XML_FORBIDDEN.search("\uffff")
    assert not _collada._XML_FORBIDDEN.search("plain \u00e9 name")


def test_module_constants() -> None:
    assert _collada.EXTENSION == ".dae"
    assert _collada.LABEL == "COLLADA"


# ---------------------------------------------------------------------------
# reference implementation
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore")
def test_reference_reads_our_file(tmp_path: Path) -> None:
    collada = pytest.importorskip("collada")
    out = tmp_path / "ref.dae"
    write_scene(_material_scene(), out)
    doc = collada.Collada(str(out))
    assert len(doc.geometries) == 1
    prims = list(doc.geometries[0].primitives)
    # The triangle joins the quad's polylist form; one block per material.
    assert [type(p).__name__ for p in prims] == ["Polylist", "Polylist"]
    assert [p.material for p in prims] == ["material0", "material1"]
    tri, quad = prims
    assert tri.vcounts.tolist() == [3] and quad.vcounts.tolist() == [4]
    np.testing.assert_array_equal(tri.vertex_index, [0, 1, 4])
    np.testing.assert_array_equal(
        quad.triangleset().vertex_index, [[0, 1, 2], [0, 2, 3]]
    )
    assert tri.normal is not None and tri.texcoordset
    assert len(doc.materials) == 2
    assert doc.materials[0].effect.diffuse is not None
    assert len(doc.scene.nodes) == 1
    geoms = list(doc.scene.objects("geometry"))
    assert len(geoms) == 1
    world = np.asarray(list(geoms[0].primitives())[0].vertex)
    expected = _surface().vertices * 2 + [1, 2, 3]
    np.testing.assert_allclose(sorted(world.tolist()), sorted(expected.tolist()))


@pytest.mark.filterwarnings("ignore")
def test_reference_file_reads_here(tmp_path: Path) -> None:
    collada = pytest.importorskip("collada")
    from collada import geometry, material, scene, source

    doc = collada.Collada()
    effect = material.Effect(
        "effect0", [], "phong", diffuse=(0.2, 0.4, 0.6), specular=(0, 1, 0)
    )
    mat = material.Material("material0", "mymaterial", effect)
    doc.effects.append(effect)
    doc.materials.append(mat)
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    normals = np.array([[0, 0, 1], [0, 0, -1]], dtype=np.float32)
    vert_src = source.FloatSource("verts", verts.ravel(), ("X", "Y", "Z"))
    norm_src = source.FloatSource("norms", normals.ravel(), ("X", "Y", "Z"))
    geom = geometry.Geometry(doc, "geometry0", "square", [vert_src, norm_src])
    inputs = source.InputList()
    inputs.addInput(0, "VERTEX", "#verts")
    inputs.addInput(1, "NORMAL", "#norms")
    idx = np.array([0, 0, 1, 0, 2, 0, 0, 1, 2, 1, 3, 1])
    triset = geom.createTriangleSet(idx, inputs, "materialref")
    geom.primitives.append(triset)
    doc.geometries.append(geom)
    matnode = scene.MaterialNode("materialref", mat, inputs=[])
    geomnode = scene.GeometryNode(geom, [matnode])
    node = scene.Node(
        "node0", children=[geomnode], transforms=[scene.TranslateTransform(0, 0, 9)]
    )
    myscene = scene.Scene("myscene", [node])
    doc.scenes.append(myscene)
    doc.scene = myscene
    path = tmp_path / "pyc.dae"
    with open(path, "wb") as fh:
        doc.write(fh)

    got = read_scene(path)
    assert got.materials[0].base_color == pytest.approx((0.2, 0.4, 0.6, 1.0))
    assert got.materials[0].extras["specular"] == pytest.approx((0.0, 1.0, 0.0, 1.0))
    mesh = got.meshes[0]
    assert len(mesh.vertices) == 6
    assert mesh.element_types.tolist() == [ELEMENT_TYPES["triangle"]] * 2
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"][[0, 3]], normals)
    assert mesh.element_attrs["material"].tolist() == [0, 0]
    np.testing.assert_array_equal(got.nodes[0].matrix[:3, 3], [0, 0, 9])
    flat = got.to_polydata()
    assert flat.vertices[:, 2].tolist() == [9.0] * 6
