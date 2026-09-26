"""COLLADA (``.dae``): the XML scene interchange format, as a SceneData.

A COLLADA document is a set of libraries - geometries, effects, materials,
images, controllers, animations, visual scenes - tied together by ``#id``
references, and a ``<scene>`` naming the visual scene to show. Each
``<geometry>`` holds a ``<mesh>`` of ``<source>`` arrays and primitive
blocks (``<triangles>``, ``<polylist>``, ``<lines>``, ...) whose ``<p>``
index tuples name one entry of each ``<input>`` per corner, so a corner
carries its own normal and texture coordinate and a vertex two corners
disagree about is split here, as OBJ readers do. ``<node>`` trees carry
transforms as ``<matrix>``, ``<translate>``, ``<rotate>`` and ``<scale>``
elements in document order, instance geometries, controllers (skins) and
other nodes; ``<animation>`` channels target a node's transform element
by the ``sid`` it declares.

Read gives a :class:`~polyxios.SceneData`: one PolyData per geometry with
``normals``, ``texcoords``, ``colors``, ``joints`` and ``weights`` as vertex
attributes and ``element_attrs["material"]`` as indices into the scene's
materials; effects as PBR-ish :class:`~polyxios.SceneMaterial` entries whose
phong pieces ride in ``extras``; nodes with their local matrix and the
transform elements they were spelled with in ``extras["transforms"]``, so an
animation still finds its target on write; skins and animations under
``global_attrs``. Write spells it all back in 1.4.1 syntax with fixed
timestamps, so the same scene always writes the same bytes.
"""

from __future__ import annotations

import base64
import dataclasses
import os
import re
from typing import Any
import warnings
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._io import Source, read_bytes, source_name, write_text
from polyxios._scene import (
    SceneData,
    SceneImage,
    SceneMaterial,
    SceneNode,
    SceneTexture,
)
from polyxios._types import PolyData
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header
from polyxios.version import version as __version__

EXTENSION: str = ".dae"
LABEL: str = "COLLADA"

_NS = "http://www.collada.org/2005/11/COLLADASchema"

# How much of the document is handed to expat at a time. Fed whole, expat
# grows one buffer to hold it and cannot grow it past 1 GiB.
_PARSE_CHUNK: int = 1 << 20

_XML_FORBIDDEN = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_NCNAME = re.compile(r"[^\W\d][\w.\-]*\Z")
# A parser folds a literal newline, tab or return in an attribute to a
# space; only the character reference survives a round trip.
_ATTR_ENTITIES = {'"': "&quot;", "\n": "&#10;", "\r": "&#13;", "\t": "&#9;"}

_UP_AXES = ("X_UP", "Y_UP", "Z_UP")

# A fixed timestamp keeps two writes of one scene byte-identical.
_EPOCH = "1970-01-01T00:00:00Z"

_TRI = ELEMENT_TYPES["triangle"]
_QUAD = ELEMENT_TYPES["quad"]
_POLYGON = ELEMENT_TYPES["polygon"]
_LINE = ELEMENT_TYPES["line"]
_POLY_LINE = ELEMENT_TYPES["poly_line"]
_STRIP = ELEMENT_TYPES["triangle_strip"]
_VERTEX = ELEMENT_TYPES["vertex"]

# Which primitive block each element type is written in.
_BLOCK_OF_TYPE = {
    _TRI: "triangles",
    _QUAD: "polylist",
    _POLYGON: "polylist",
    _LINE: "lines",
    _POLY_LINE: "linestrips",
    _STRIP: "tristrips",
}

_BLOCKS = ("triangles", "polylist", "lines", "linestrips", "tristrips")
_BLOCK_INDEX = {code: _BLOCKS.index(block) for code, block in _BLOCK_OF_TYPE.items()}

_WRAP_TO_GL = {
    "WRAP": 10497,
    "MIRROR": 33648,
    "CLAMP": 33071,
    "BORDER": 33069,
    "NONE": 33071,
}
_GL_TO_WRAP = {10497: "WRAP", 33648: "MIRROR", 33071: "CLAMP", 33069: "BORDER"}
_FILTER_TO_GL = {
    "NEAREST": 9728,
    "LINEAR": 9729,
    "NEAREST_MIPMAP_NEAREST": 9984,
    "LINEAR_MIPMAP_NEAREST": 9985,
    "NEAREST_MIPMAP_LINEAR": 9986,
    "LINEAR_MIPMAP_LINEAR": 9987,
}
_GL_TO_FILTER = {v: k for k, v in _FILTER_TO_GL.items()}

_MEDIA_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tga": "image/x-tga",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "webp": "image/webp",
    "ktx": "image/ktx",
    "ktx2": "image/ktx2",
    "dds": "image/vnd-ms.dds",
}
_EXT_OF_MEDIA = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif"}

_PATH_OF_KIND = {
    "matrix": "matrix",
    "translate": "translation",
    "rotate": "rotation",
    "scale": "scale",
    "lookat": "lookat",
}
_KIND_OF_PATH = {v: k for k, v in _PATH_OF_KIND.items()}
_KIND_SIZE = {"matrix": 16, "translate": 3, "rotate": 4, "scale": 3, "lookat": 9}

_PARAM_WIDTH = {
    "float2": 2,
    "float3": 3,
    "float4": 4,
    "float2x2": 4,
    "float3x3": 9,
    "float4x4": 16,
}

_SHADINGS = ("phong", "lambert", "blinn", "constant")
_OPAQUE_MODES = ("A_ONE", "A_ZERO", "RGB_ZERO", "RGB_ONE")
_INTERPOLATIONS = ("LINEAR", "BEZIER", "HERMITE", "CARDINAL", "BSPLINE", "STEP")
_MATERIAL_TEXTURES = ("base_color_texture", "normal_texture", "emissive_texture")

_IDENTITY = np.eye(4, dtype=np.float64)

# Warnings are attributed to the caller's line, whatever depth they come from.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# read
# =============================================================================


def read_scene(path: Source, **opts: Any) -> SceneData:
    """Read a COLLADA document and return its scene.

    Parameters
    ----------
    path
        Path or open binary file object of a ``.dae`` document.
    **opts
        Accepted for interface symmetry and ignored.

    Returns
    -------
    SceneData
        One mesh per ``<geometry>``, in library order (every
        ``<library_geometries>`` of the document, as with the other
        libraries), its ``name`` as ``global_attrs["mesh_name"]``, the
        visual scenes' node trees with the instanced one active, effects as
        materials with their textures and images, skins and animations under
        ``global_attrs``. COLLADA has no alpha mask, so ``alpha_mode`` is
        ``BLEND`` when the material's alpha is below one and ``OPAQUE``
        otherwise.

    Raises
    ------
    CodecError
        When the document is not well-formed XML, its root is not
        ``<COLLADA>`` of version 1.4 or 1.5, an array's ``count`` disagrees
        with its content, an accessor reads past its array, an index tuple
        names an entry a source lacks, a ``<p>`` or ``<vcount>`` length
        disagrees with the block's ``count``, an input inside ``<vertices>``
        has a row count other than the positions', a ``#url`` names an id
        the document does not hold or an ``<instance_node>`` names one that
        is not a node, a transform element has the wrong number of values,
        a node instances itself, or ``up_axis`` is not one of the three the
        specification allows.
    """
    name = source_name(path)
    raw = read_bytes(path)
    root = _parse(raw, name)
    doc = _Doc(root=root, name=name, size=len(raw), ids=_ids(root))

    global_attrs: dict[str, Any] = {}
    asset = _read_asset(doc)
    if asset:
        global_attrs["asset"] = asset

    images = _read_images(doc)
    image_index: dict[str, int] = {}
    for i, img_id in enumerate(images[1]):
        if img_id:
            image_index.setdefault(img_id, i)
    tex = _Textures(image_index=image_index)
    materials, material_index = _read_materials(doc, tex)

    geometries = _library(root, "library_geometries", "geometry")
    meshes: list[_Mesh | None] = [None] * len(geometries)
    slot_of_gid = {}
    for i, geo in enumerate(geometries):
        gid = geo.get("id")
        if gid is not None:
            slot_of_gid[gid] = i
    st = _State(
        doc=doc,
        geometries=geometries,
        slot_of_gid=slot_of_gid,
        meshes=meshes,
        material_index=material_index,
    )
    scenes, active, scene_name = _read_scenes(st)

    for i in range(len(meshes)):
        _mesh_at(st, i, None)
        _resolve_deferred(st, i)
    if not st.nodes:
        st.nodes = [SceneNode(mesh=i) for i in range(len(meshes))]
        scenes = (tuple(range(len(meshes))),) if meshes else ()
    polys = tuple(_finish_mesh(st, m) for m in meshes if m is not None)

    if st.skins:
        _resolve_joints(st)
        global_attrs["skins"] = st.skins
    animations = _read_animations(st)
    if animations:
        global_attrs["animations"] = animations

    return SceneData(
        meshes=polys,
        nodes=tuple(st.nodes),
        materials=materials,
        textures=tuple(tex.textures),
        images=images[0],
        scenes=scenes,
        active_scene=active,
        name=scene_name,
        global_attrs=global_attrs,
    )


def read(path: Source, *, lazy: bool = False, **opts: Any) -> PolyData:
    """Read a COLLADA document flattened to one PolyData.

    Parameters
    ----------
    path
        Path or open binary file object of a ``.dae`` document.
    lazy
        Not supported: the geometry is XML text.
    **opts
        Accepted for interface symmetry and ignored.

    Returns
    -------
    PolyData
        Every instanced mesh of the active scene under its node's world
        transform, merged.

    Warns
    -----
    UserWarning
        Always: the scene graph, materials and animations are dropped;
        :func:`read_scene` keeps them.

    Raises
    ------
    LazyReadError
        If ``lazy=True``.
    CodecError
        Whatever :func:`read_scene` refuses.
    """
    if lazy:
        raise LazyReadError("COLLADA is XML text and cannot be memory-mapped.")
    warnings.warn(
        f"'{source_name(path)}' is a scene format (COLLADA): read() flattens "
        "the scene graph, materials, skins and animations into a single "
        "PolyData. Use polyxios.read_scene() to preserve the full scene.",
        stacklevel=3,  # user -> read -> here
    )
    return read_scene(path).to_polydata()


# -----------------------------------------------------------------------------
# document plumbing
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class _Doc:
    """The parsed document and the lookups every phase needs."""

    root: ET.Element
    name: str
    size: int
    ids: dict[str, ET.Element]
    arrays: dict[int, np.ndarray | list[str]] = dataclasses.field(default_factory=dict)
    warned: set[str] = dataclasses.field(default_factory=set)
    handed: set[int] = dataclasses.field(default_factory=set)


def _parse(raw: bytes, name: str) -> ET.Element:
    parser = ET.XMLParser()
    try:
        for start in range(0, len(raw), _PARSE_CHUNK):
            parser.feed(raw[start : start + _PARSE_CHUNK])
        root = parser.close()
    except ET.ParseError as exc:
        raise CodecError(f"{name!r} is not well-formed XML: {exc}") from exc
    if _local(root.tag) != "COLLADA":
        raise CodecError(
            f"{name!r}: the document root is <{_local(root.tag)}>, not <COLLADA>."
        )
    version = root.get("version", "")
    if not version.startswith(("1.4", "1.5")):
        raise CodecError(
            f"{name!r}: COLLADA version {version!r} is not one this reader "
            "knows (1.4 or 1.5)."
        )
    return root


def _ids(root: ET.Element) -> dict[str, ET.Element]:
    ids: dict[str, ET.Element] = {}
    for elem in root.iter():
        eid = elem.get("id")
        if eid is not None:
            ids.setdefault(eid, elem)
    return ids


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(elem: ET.Element | None, tag: str) -> list[ET.Element]:
    if elem is None:
        return []
    return [child for child in elem if _local(child.tag) == tag]


def _child(elem: ET.Element | None, tag: str) -> ET.Element | None:
    if elem is None:
        return None
    for child in elem:
        if _local(child.tag) == tag:
            return child
    return None


def _library(root: ET.Element, library: str, tag: str) -> list[ET.Element]:
    """Return the ``tag`` entries of every ``<library>`` of the root, in document order."""
    return [entry for lib in _children(root, library) for entry in _children(lib, tag)]


def _ref(doc: _Doc, url: str | None, what: str) -> ET.Element:
    """Return the element a ``#id`` reference names."""
    if url is None:
        raise CodecError(f"{doc.name!r}: {what} has no url.")
    if not url.startswith("#"):
        raise CodecError(
            f"{doc.name!r}: {what} names {url!r}, an external document, "
            "which this reader does not follow."
        )
    elem = doc.ids.get(url[1:])
    if elem is None:
        raise CodecError(
            f"{doc.name!r}: {what} names {url!r}, which the document does not hold."
        )
    return elem


def _int(text: str | None, what: str, *, default: int | None = None) -> int:
    if text is None:
        if default is None:
            raise CodecError(f"{what} is missing.")
        return default
    try:
        value = int(text)
    except ValueError as exc:
        raise CodecError(f"{what} is {text!r}, not an integer.") from exc
    if value < 0:
        raise CodecError(f"{what} is {value}, a negative count.")
    return value


def _ints(text: str | None, what: str) -> np.ndarray:
    try:
        return np.array((text or "").split(), dtype=np.int64)
    except ValueError as exc:
        raise CodecError(f"{what} is not integers.") from exc


def _floats(text: str | None, what: str, n: int | None = None) -> np.ndarray:
    values = (text or "").split()
    if n is not None and len(values) != n:
        raise CodecError(f"{what} needs {n} numbers, got {len(values)}.")
    try:
        return np.array(values, dtype=np.float64)
    except ValueError as exc:
        raise CodecError(f"{what} is not numbers: {(text or '')[:60]!r}.") from exc


def _array(doc: _Doc, elem: ET.Element, what: str) -> np.ndarray | list[str]:
    """Return the values of a ``<float_array>``, ``<int_array>`` or name array."""
    cached = doc.arrays.get(id(elem))
    if cached is not None:
        return cached
    kind = _local(elem.tag)
    count = _int(elem.get("count"), f"{doc.name!r}: the count of {what}")
    if count > doc.size:
        raise CodecError(
            f"{doc.name!r}: {what} declares count={count}, more values than "
            f"the file has bytes ({doc.size})."
        )
    tokens = (elem.text or "").split()
    if len(tokens) != count:
        raise CodecError(
            f"{doc.name!r}: {what} declares count={count} but holds "
            f"{len(tokens)} values."
        )
    values: np.ndarray | list[str]
    if kind == "float_array":
        try:
            values = np.array(tokens, dtype=np.float64)
        except ValueError as exc:
            raise CodecError(
                f"{doc.name!r}: {what} is not numbers: {(elem.text or '')[:60]!r}."
            ) from exc
    elif kind == "int_array":
        try:
            values = np.array(tokens, dtype=np.int64)
        except ValueError as exc:
            raise CodecError(
                f"{doc.name!r}: {what} is not integers: {(elem.text or '')[:60]!r}."
            ) from exc
    else:
        values = tokens
    doc.arrays[id(elem)] = values
    return values


def _source(doc: _Doc, src: ET.Element, what: str) -> np.ndarray | list[str]:
    """Return a ``<source>`` as its accessor reads it: floats ``(count, k)`` or names."""
    sid = src.get("id", "?")
    acc = _child(_child(src, "technique_common"), "accessor")
    if acc is None:
        raise CodecError(f"{doc.name!r}: source '{sid}' ({what}) has no <accessor>.")
    arr_elem = _ref(doc, acc.get("source"), f"the accessor of source '{sid}'")
    count = _int(acc.get("count"), f"{doc.name!r}: the count of accessor '{sid}'")
    stride = _int(acc.get("stride"), f"{doc.name!r}: stride of '{sid}'", default=1)
    offset = _int(acc.get("offset"), f"{doc.name!r}: offset of '{sid}'", default=0)
    if stride == 0:
        raise CodecError(f"{doc.name!r}: the accessor of source '{sid}' has stride 0.")
    values = _array(doc, arr_elem, f"the array of source '{sid}'")
    need = offset + count * stride
    if need > len(values):
        raise CodecError(
            f"{doc.name!r}: the accessor of source '{sid}' reads {need} values "
            f"from an array of {len(values)}."
        )
    used: list[int] = []
    at = 0
    for p in _children(acc, "param"):
        width = _PARAM_WIDTH.get(p.get("type", ""), 1)
        if p.get("name"):
            used.extend(range(at, at + width))
        at += width
    if isinstance(values, list):
        rows = [values[offset + i * stride] for i in range(count)]
        return rows
    table = values[offset:need].reshape(count, stride)
    if used and max(used) < stride and len(used) != stride:
        table = table[:, used]
    return table


def _float_source(doc: _Doc, src: ET.Element, what: str) -> np.ndarray:
    table = _source(doc, src, what)
    if isinstance(table, list):
        raise CodecError(
            f"{doc.name!r}: source '{src.get('id', '?')}' ({what}) holds names, "
            "not numbers."
        )
    return table


def _name_source(doc: _Doc, src: ET.Element, what: str) -> list[str]:
    table = _source(doc, src, what)
    if not isinstance(table, list):
        raise CodecError(
            f"{doc.name!r}: source '{src.get('id', '?')}' ({what}) holds numbers, "
            "not names."
        )
    return table


def _warn(message: str) -> None:
    warnings.warn(message, skip_file_prefixes=(_PACKAGE_DIR,))


# -----------------------------------------------------------------------------
# asset, images, effects, materials
# -----------------------------------------------------------------------------


def _read_asset(doc: _Doc) -> dict[str, Any]:
    asset = _child(doc.root, "asset")
    out: dict[str, Any] = {}
    if asset is None:
        return out
    up = _child(asset, "up_axis")
    if up is not None:
        value = (up.text or "").strip()
        if value not in _UP_AXES:
            raise CodecError(
                f"{doc.name!r}: up_axis is {value!r}; the specification allows "
                f"{', '.join(_UP_AXES)}."
            )
        out["up_axis"] = value
    unit = _child(asset, "unit")
    if unit is not None:
        spelled = unit.get("meter", "1")
        if "," in spelled and "." not in spelled:
            # Some exporters write the unit in the machine's locale.
            _warn(
                f"{doc.name!r}: the unit's meter is spelled {spelled!r} with a "
                "decimal comma; it is read as a point."
            )
            spelled = spelled.replace(",", ".")
        meter = _floats(spelled, f"{doc.name!r}: the unit's meter", 1)
        out["unit"] = {"name": unit.get("name", "meter"), "meter": float(meter[0])}
    tool = _child(_child(asset, "contributor"), "authoring_tool")
    if tool is not None and tool.text:
        out["authoring_tool"] = tool.text.strip()
    for key in ("created", "modified"):
        elem = _child(asset, key)
        if elem is not None and elem.text:
            out[key] = elem.text.strip()
    return out


def _media_type(uri: str) -> str | None:
    ext = uri.rsplit(".", 1)[-1].lower() if "." in uri else ""
    return _MEDIA_TYPES.get(ext)


def _read_images(doc: _Doc) -> tuple[tuple[SceneImage, ...], list[str]]:
    images: list[SceneImage] = []
    ids: list[str] = []
    for img in _library(doc.root, "library_images", "image"):
        init = _child(img, "init_from")
        name = img.get("name", "")
        image = SceneImage(name=name)
        if init is not None:
            ref = _child(init, "ref")
            hexed = _child(init, "hex")
            if hexed is not None:
                try:
                    data = bytes.fromhex("".join((hexed.text or "").split()))
                except ValueError as exc:
                    raise CodecError(
                        f"{doc.name!r}: image '{img.get('id', '?')}' holds "
                        f"<hex> that is not hexadecimal: {exc}"
                    ) from exc
                fmt = hexed.get("format", "").lower()
                image = SceneImage(
                    data=data, media_type=_MEDIA_TYPES.get(fmt), name=name
                )
            else:
                uri = ((ref.text if ref is not None else init.text) or "").strip()
                image = _image_of_uri(doc, uri, name, img.get("id", "?"))
        images.append(image)
        ids.append(img.get("id", ""))
    return tuple(images), ids


def _image_of_uri(doc: _Doc, uri: str, name: str, img_id: str) -> SceneImage:
    if uri.startswith("data:"):
        header, sep, payload = uri[5:].partition(",")
        if not sep:
            raise CodecError(
                f"{doc.name!r}: image '{img_id}' has a data URI without a comma."
            )
        media, _, encoding = header.partition(";")
        if encoding != "base64":
            raise CodecError(
                f"{doc.name!r}: image '{img_id}' has a data URI that is not base64."
            )
        try:
            data = base64.b64decode(payload, validate=True)
        except ValueError as exc:
            raise CodecError(
                f"{doc.name!r}: image '{img_id}' has a data URI that does not decode: {exc}"
            ) from exc
        return SceneImage(data=data, media_type=media or None, name=name)
    return SceneImage(uri=uri, media_type=_media_type(uri), name=name)


def _color_of(elem: ET.Element | None, what: str) -> tuple[float, ...] | None:
    color = _child(elem, "color")
    if color is None:
        return None
    values = _floats(color.text, what)
    if len(values) not in (3, 4):
        raise CodecError(f"{what} needs 3 or 4 numbers, got {len(values)}.")
    if len(values) == 3:
        values = np.append(values, 1.0)
    return tuple(float(v) for v in values)


def _float_of(elem: ET.Element | None, what: str) -> float | None:
    f = _child(elem, "float")
    if f is None:
        return None
    return float(_floats(f.text, what, 1)[0])


@dataclasses.dataclass
class _Textures:
    """The texture table a read builds, deduplicated by settings."""

    image_index: dict[str, int]
    textures: list[SceneTexture] = dataclasses.field(default_factory=list)
    index: dict[SceneTexture, int] = dataclasses.field(default_factory=dict)


def _texture_of(
    doc: _Doc,
    effect: ET.Element,
    params: dict[str | None, ET.Element],
    elem: ET.Element | None,
    tex_table: _Textures,
) -> int | None:
    """Resolve a ``<texture texture=sid>`` through sampler and surface to a texture index."""
    tex = _child(elem, "texture")
    if tex is None:
        return None
    ref = tex.get("texture", "")
    sampler = _child(params.get(ref), "sampler2D")
    image_id = ref
    settings: dict[str, Any] = {}
    if sampler is not None:
        surface_sid = (
            (_child(sampler, "source").text or "").strip()
            if _child(sampler, "source") is not None
            else ""
        )
        surface = _child(params.get(surface_sid), "surface")
        init = _child(surface, "init_from")
        image_id = (init.text or "").strip() if init is not None else ""
        inst = _child(sampler, "instance_image")
        if inst is not None:
            image_id = inst.get("url", "").removeprefix("#")
        for key, table in (("wrap_s", _WRAP_TO_GL), ("wrap_t", _WRAP_TO_GL)):
            node = _child(sampler, key)
            if node is not None and (node.text or "").strip() in table:
                settings[key] = table[(node.text or "").strip()]
        for key, tag in (("min_filter", "minfilter"), ("mag_filter", "magfilter")):
            node = _child(sampler, tag)
            if node is not None and (node.text or "").strip() in _FILTER_TO_GL:
                settings[key] = _FILTER_TO_GL[(node.text or "").strip()]
    if image_id not in tex_table.image_index:
        _warn(
            f"{doc.name!r}: effect '{effect.get('id', '?')}' names texture "
            f"{ref!r}, which reaches no image; the texture is dropped."
        )
        return None
    texture = SceneTexture(image=tex_table.image_index[image_id], **settings)
    slot = tex_table.index.get(texture)
    if slot is None:
        slot = len(tex_table.textures)
        tex_table.textures.append(texture)
        tex_table.index[texture] = slot
    return slot


def _luminance(rgb: tuple[float, ...]) -> float:
    return 0.212671 * rgb[0] + 0.715160 * rgb[1] + 0.072169 * rgb[2]


def _read_effect(
    doc: _Doc, effect: ET.Element, name: str, tex_table: _Textures
) -> SceneMaterial:
    profile = _child(effect, "profile_COMMON")
    technique = _child(profile, "technique")
    shading_elem = None
    for child in technique if technique is not None else ():
        if _local(child.tag) in _SHADINGS:
            shading_elem = child
            break
    what = f"{doc.name!r}: effect '{effect.get('id', '?')}'"
    extras: dict[str, Any] = {
        "shading": _local(shading_elem.tag) if shading_elem is not None else "phong"
    }
    if shading_elem is None:
        return SceneMaterial(name=name, metallic=0.0, extras=extras)

    params = {p.get("sid"): p for p in effect.iter() if _local(p.tag) == "newparam"}
    diffuse = _child(shading_elem, "diffuse")
    base = _color_of(diffuse, f"{what} diffuse") or (1.0, 1.0, 1.0, 1.0)
    base_tex = _texture_of(doc, effect, params, diffuse, tex_table)
    emission = _child(shading_elem, "emission")
    emissive = (_color_of(emission, f"{what} emission") or (0.0, 0.0, 0.0, 1.0))[:3]
    emissive_tex = _texture_of(doc, effect, params, emission, tex_table)
    for key in ("ambient", "specular", "reflective"):
        color = _color_of(_child(shading_elem, key), f"{what} {key}")
        if color is not None:
            extras[key] = color
    for key in ("shininess", "reflectivity", "index_of_refraction"):
        value = _float_of(_child(shading_elem, key), f"{what} {key}")
        if value is not None:
            extras[key] = value

    transparent = _child(shading_elem, "transparent")
    mode = transparent.get("opaque", "A_ONE") if transparent is not None else "A_ONE"
    if mode not in _OPAQUE_MODES:
        _warn(
            f"{what} has transparent opaque={mode!r}, which the specification does "
            "not name; it is read as A_ONE."
        )
        mode = "A_ONE"
    tcolor = _color_of(transparent, f"{what} transparent")
    amount = _float_of(_child(shading_elem, "transparency"), f"{what} transparency")
    alpha = base[3]
    if transparent is not None or amount is not None:
        amount = 1.0 if amount is None else amount
        if tcolor is None:
            tcolor = (0.0, 0.0, 0.0, 1.0)
        if mode == "A_ONE" and amount == 0.0 and tcolor == (1.0, 1.0, 1.0, 1.0):
            # SketchUp, Google Earth and Blender before 2.8 spell an opaque
            # material this way; read literally it would be invisible.
            if "opaque" not in doc.warned:
                doc.warned.add("opaque")
                _warn(
                    f"{what} has A_ONE transparency 0 with a white transparent "
                    "colour, the exporter convention for opaque; it and any "
                    "other such effect of the document are read as opaque."
                )
            amount = 1.0
        if mode == "A_ONE":
            alpha = tcolor[3] * amount
        elif mode == "A_ZERO":
            alpha = 1.0 - tcolor[3] * amount
        elif mode == "RGB_ZERO":
            alpha = 1.0 - _luminance(tcolor) * amount
        elif mode == "RGB_ONE":
            alpha = _luminance(tcolor) * amount
        alpha = min(max(alpha * base[3], 0.0), 1.0)
        extras["transparent_mode"] = mode

    normal_tex = None
    for extra in effect.iter():
        if _local(extra.tag) == "bump":
            normal_tex = _texture_of(doc, effect, params, extra, tex_table)
            break
    double_sided = any(
        _local(e.tag) == "double_sided" and (e.text or "").strip() in ("1", "true")
        for e in effect.iter()
    )
    return SceneMaterial(
        name=name,
        base_color=(base[0], base[1], base[2], alpha),
        metallic=0.0,
        roughness=1.0,
        emissive=tuple(emissive),
        alpha_mode="BLEND" if alpha < 1.0 else "OPAQUE",
        double_sided=double_sided,
        base_color_texture=base_tex,
        normal_texture=normal_tex,
        emissive_texture=emissive_tex,
        extras=extras,
    )


def _read_materials(
    doc: _Doc, tex_table: _Textures
) -> tuple[tuple[SceneMaterial, ...], dict[str, int]]:
    materials: list[SceneMaterial] = []
    index: dict[str, int] = {}
    for mat in _library(doc.root, "library_materials", "material"):
        mid = mat.get("id", "")
        inst = _child(mat, "instance_effect")
        name = mat.get("name", "")
        if inst is None:
            materials.append(
                SceneMaterial(name=name, metallic=0.0, extras={"shading": "phong"})
            )
        else:
            effect = _ref(doc, inst.get("url"), f"material '{mid}'")
            materials.append(_read_effect(doc, effect, name, tex_table))
        if mid:
            index.setdefault(mid, len(materials) - 1)
    return tuple(materials), index


# -----------------------------------------------------------------------------
# geometry
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class _Mesh:
    """One geometry, read once and shared by every node instancing it."""

    poly: PolyData
    positions_of: np.ndarray
    n_positions: int
    symbols: list[tuple[str | None, int]]
    gid: str
    materials: np.ndarray | None = None
    bound: dict[str, int] | None = None
    skinned: bool = False


@dataclasses.dataclass
class _State:
    """What the node walk accumulates."""

    doc: _Doc
    geometries: list[ET.Element]
    slot_of_gid: dict[str, int]
    meshes: list[_Mesh | None]
    material_index: dict[str, int]
    nodes: list[SceneNode | None] = dataclasses.field(default_factory=list)
    index_by_id: dict[str, int] = dataclasses.field(default_factory=dict)
    index_by_sid: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    skins: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    skin_of_ctrl: dict[str, int] = dataclasses.field(default_factory=dict)
    pending_joints: list[tuple[int, list[str], list[str]]] = dataclasses.field(
        default_factory=list
    )


def _prim_rows(
    doc: _Doc, prim: ET.Element, stride: int, gid: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (corners, sizes, types) for one primitive block."""
    tag = _local(prim.tag)
    what = f"{doc.name!r}: <{tag}> of geometry '{gid}'"
    count = _int(prim.get("count"), f"{what} count")

    def ints(text: str | None, label: str) -> np.ndarray:
        return _ints(text, f"{what}: {label}")

    p_elem = _child(prim, "p")
    if tag in ("triangles", "lines"):
        per = 3 if tag == "triangles" else 2
        p = ints(p_elem.text if p_elem is not None else "", "<p>")
        if len(p) != count * per * stride:
            raise CodecError(
                f"{what} declares count={count}, which needs {count * per * stride} "
                f"indices, but its <p> holds {len(p)}."
            )
        sizes = np.full(count, per, dtype=np.int64)
        types = np.full(count, _TRI if per == 3 else _LINE, dtype=np.uint8)
        return p.reshape(-1, stride), sizes, types

    if tag == "polylist":
        vc = _child(prim, "vcount")
        vcount = ints(vc.text if vc is not None else "", "<vcount>")
        if len(vcount) != count:
            raise CodecError(
                f"{what} declares count={count} but its <vcount> holds {len(vcount)}."
            )
        if len(vcount) and vcount.min() < 1:
            raise CodecError(f"{what} has a <vcount> entry below 1.")
        p = ints(p_elem.text if p_elem is not None else "", "<p>")
        need = int(vcount.sum()) * stride
        if len(p) != need:
            raise CodecError(
                f"{what}: <vcount> sums to {need} indices but <p> holds {len(p)}."
            )
        return p.reshape(-1, stride), vcount, _types_of_sizes(vcount)

    blocks: list[np.ndarray] = []
    holes = False
    for child in prim:
        kind = _local(child.tag)
        if kind == "p":
            blocks.append(ints(child.text, "<p>"))
        elif kind == "ph":
            outer = _child(child, "p")
            if outer is None:
                raise CodecError(f"{what} has a <ph> without a <p>.")
            holes = True
            blocks.append(ints(outer.text, "<ph><p>"))
    if len(blocks) != count:
        raise CodecError(
            f"{what} declares count={count} but holds {len(blocks)} <p> blocks."
        )
    if holes:
        _warn(f"{what} has polygons with holes; the holes are dropped.")
    for block in blocks:
        if len(block) % stride:
            raise CodecError(
                f"{what}: a <p> of {len(block)} indices is not a whole number of "
                f"{stride}-index corners."
            )
    corners = [b.reshape(-1, stride) for b in blocks]
    sizes = np.array([len(c) for c in corners], dtype=np.int64)
    if tag == "polygons":
        if len(sizes) and sizes.min() < 1:
            raise CodecError(f"{what} has an empty <p>.")
        types = _types_of_sizes(sizes)
    elif tag == "linestrips":
        if len(sizes) and sizes.min() < 2:
            raise CodecError(f"{what} has a strip of fewer than 2 corners.")
        types = np.full(len(sizes), _POLY_LINE, dtype=np.uint8)
    elif tag == "tristrips":
        if len(sizes) and sizes.min() < 3:
            raise CodecError(f"{what} has a strip of fewer than 3 corners.")
        types = np.full(len(sizes), _STRIP, dtype=np.uint8)
    else:  # trifans
        if len(sizes) and sizes.min() < 3:
            raise CodecError(f"{what} has a fan of fewer than 3 corners.")
        fans = []
        for c in corners:
            n = len(c) - 2
            fans.append(
                np.stack([np.repeat(c[:1], n, 0), c[1:-1], c[2:]], 1).reshape(
                    -1, stride
                )
            )
        corners = fans
        sizes = np.full(sum(len(f) // 3 for f in fans), 3, dtype=np.int64)
        types = np.full(len(sizes), _TRI, dtype=np.uint8)
    stacked = np.concatenate(corners) if corners else np.zeros((0, stride), np.int64)
    return stacked, sizes, types


def _types_of_sizes(sizes: np.ndarray) -> np.ndarray:
    types = np.full(len(sizes), _POLYGON, dtype=np.uint8)
    types[sizes == 3] = _TRI
    types[sizes == 4] = _QUAD
    types[sizes == 2] = _LINE
    types[sizes == 1] = _VERTEX
    return types


def _attr_name(semantic: str, set_no: str) -> str:
    if semantic == "NORMAL":
        return "normals"
    if semantic == "COLOR":
        return "colors" if set_no in ("", "0") else f"colors_{set_no}"
    if semantic == "TEXCOORD":
        return "texcoords" if set_no in ("", "0") else f"texcoords_{set_no}"
    base = semantic.lower()
    return base if set_no in ("", "0") else f"{base}_{set_no}"


def _read_geometry(doc: _Doc, geo: ET.Element, slot: int) -> _Mesh:
    gid = geo.get("id") or f"geometry_{slot}"
    label = geo.get("name")
    global_attrs = {"mesh_name": label} if label else {}
    mesh = _child(geo, "mesh")
    if mesh is None:
        kinds = [_local(c.tag) for c in geo if _local(c.tag) not in ("asset", "extra")]
        _warn(
            f"{doc.name!r}: geometry '{gid}' holds <{kinds[0] if kinds else '?'}>, "
            "which this reader does not read; it is an empty mesh."
        )
        empty = dataclasses.replace(_empty(), global_attrs=global_attrs)
        return _Mesh(empty, np.zeros(0, np.int64), 0, [], gid)

    vertices = _child(mesh, "vertices")
    if vertices is None:
        raise CodecError(f"{doc.name!r}: geometry '{gid}' has no <vertices>.")
    positions = None
    per_position: list[tuple[str, np.ndarray]] = []
    source_of: dict[str, int] = {}
    for inp in _children(vertices, "input"):
        semantic = inp.get("semantic", "")
        src = _ref(doc, inp.get("source"), f"the {semantic} input of '{gid}'")
        table = _float_source(doc, src, f"{semantic} of '{gid}'")
        if semantic == "POSITION":
            if table.shape[1] != 3:
                raise CodecError(
                    f"{doc.name!r}: geometry '{gid}' has POSITION with "
                    f"{table.shape[1]} components, not 3."
                )
            positions = table
            source_of[""] = id(src)
        else:
            attr = _attr_name(semantic, inp.get("set", ""))
            if attr not in source_of:
                per_position.append((attr, table))
                source_of[attr] = id(src)
    if positions is None:
        raise CodecError(f"{doc.name!r}: geometry '{gid}' has no POSITION input.")
    n_pos = len(positions)
    for attr, table in per_position:
        if len(table) != n_pos:
            raise CodecError(
                f"{doc.name!r}: geometry '{gid}' gives {attr} inside <vertices> "
                f"with {len(table)} rows for {n_pos} positions."
            )

    # One column per distinct (semantic, set, source): two blocks naming the
    # same semantic from different sources index different tables.
    columns: list[tuple[str, np.ndarray]] = [("", positions)]
    column_of: dict[tuple[str, str, int], int] = {}
    corner_blocks: list[np.ndarray] = []
    sizes_blocks: list[np.ndarray] = []
    types_blocks: list[np.ndarray] = []
    symbols: list[tuple[str | None, int]] = []
    n_tokens = 0
    for prim in mesh:
        tag = _local(prim.tag)
        if tag not in (
            "triangles",
            "polylist",
            "polygons",
            "lines",
            "linestrips",
            "tristrips",
            "trifans",
        ):
            continue
        what = f"{doc.name!r}: <{tag}> of geometry '{gid}'"
        inputs = _children(prim, "input")
        offsets = [
            _int(i.get("offset"), f"{what} input offset", default=0) for i in inputs
        ]
        stride = max(offsets, default=-1) + 1
        vertex_input = None
        mapping: list[tuple[int, int, int]] = []  # (p column, mesh column, count)
        for inp, offset in zip(inputs, offsets, strict=True):
            semantic = inp.get("semantic", "")
            if semantic == "VERTEX":
                src = _ref(doc, inp.get("source"), f"{what} VERTEX input")
                if src is not vertices:
                    raise CodecError(
                        f"{what} names {inp.get('source')!r} as VERTEX, which is not "
                        "the mesh's <vertices>."
                    )
                vertex_input = offset
                mapping.append((offset, 0, n_pos))
                continue
            src = _ref(doc, inp.get("source"), f"{what} {semantic} input")
            key = (semantic, inp.get("set", ""), id(src))
            attr = _attr_name(semantic, inp.get("set", ""))
            col = column_of.get(key)
            if col is None:
                table = _float_source(doc, src, f"{semantic} of '{gid}'")
                if stride == 1 and len(table) == n_pos:
                    # Every input shares the one index, so this is a value per
                    # position, and the vertices stay the positions themselves.
                    # A later block naming another source for the same
                    # attribute takes the column path and splits its corners.
                    known = source_of.get(attr)
                    if known is None:
                        per_position.append((attr, table))
                        source_of[attr] = id(src)
                    if known is None or known == id(src):
                        continue
                columns.append((attr, table))
                col = len(columns) - 1
                column_of[key] = col
            mapping.append((offset, col, len(columns[col][1])))
        if vertex_input is None:
            raise CodecError(f"{what} has no VERTEX input.")
        corners, sizes, types = _prim_rows(doc, prim, stride, gid)
        n_tokens += corners.size
        wide = np.full((len(corners), len(columns)), -1, dtype=np.int64)
        for p_col, mesh_col, limit in mapping:
            idx = corners[:, p_col]
            if len(idx) and (idx.min() < 0 or idx.max() >= limit):
                bad = int(idx.min()) if idx.min() < 0 else int(idx.max())
                raise CodecError(
                    f"{what} names entry {bad} of a source holding {limit} "
                    f"({columns[mesh_col][0] or 'POSITION'})."
                )
            wide[:, mesh_col] = idx
        corner_blocks.append(wide)
        sizes_blocks.append(sizes)
        types_blocks.append(types)
        symbols.append((prim.get("material"), len(sizes)))

    n_elements = sum(len(s) for s in sizes_blocks)
    validate_header(n_pos, n_elements, n_tokens, doc.size)

    if corner_blocks:
        width = max(b.shape[1] for b in corner_blocks)
        all_corners = np.concatenate(
            [
                np.pad(b, ((0, 0), (0, width - b.shape[1])), constant_values=-1)
                for b in corner_blocks
            ]
        )
        sizes = np.concatenate(sizes_blocks)
        types = np.concatenate(types_blocks)
    else:
        all_corners = np.zeros((0, len(columns)), dtype=np.int64)
        sizes = np.zeros(0, dtype=np.int64)
        types = np.zeros(0, dtype=np.uint8)

    if len(columns) == 1:
        connectivity = all_corners[:, 0]
        positions_of = np.arange(n_pos, dtype=np.int64)
        verts = _own(doc, source_of[""], positions)
        attrs = {name: _own(doc, source_of[name], t) for name, t in per_position}
    else:
        unique, first, inverse = np.unique(
            all_corners, axis=0, return_index=True, return_inverse=True
        )
        order = np.argsort(first, kind="stable")
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order))
        unique = unique[order]
        connectivity = rank[inverse.ravel()]
        positions_of = unique[:, 0]
        verts = positions[positions_of]
        attrs = {name: table[positions_of] for name, table in per_position}
        covered: dict[str, np.ndarray] = {}
        for col in range(1, len(columns)):
            name, table = columns[col]
            idx = unique[:, col]
            have = idx >= 0
            if name in attrs:
                previous = attrs[name]
                if previous.shape[1] != table.shape[1]:
                    _warn(
                        f"{doc.name!r}: geometry '{gid}' gives {name} two widths; "
                        "the later source is dropped."
                    )
                else:
                    previous[have] = table[idx[have]]
            else:
                values = np.zeros((len(unique), table.shape[1]), dtype=np.float64)
                values[have] = table[idx[have]]
                attrs[name] = values
                covered[name] = np.zeros(len(unique), dtype=bool)
            if name in covered:
                covered[name] |= have
        partial = [name for name, seen in covered.items() if not seen.all()]
        if partial:
            _warn(
                f"{doc.name!r}: geometry '{gid}' gives {partial} in some primitive "
                "blocks but not others; the corners without one are zero."
            )

    offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    poly = PolyData(
        vertices=np.ascontiguousarray(verts, dtype=np.float64),
        connectivity=connectivity.astype(np.int32),
        offsets=offsets.astype(np.int32),
        element_types=types,
        vertex_attrs={k: np.ascontiguousarray(v) for k, v in attrs.items()},
        global_attrs=global_attrs,
    )
    return _Mesh(poly, positions_of, n_pos, symbols, gid)


def _own(doc: _Doc, key: int, table: np.ndarray) -> np.ndarray:
    """Return ``table`` as is the first time source ``key`` is handed out, a copy after.

    A single-column geometry keeps its ``<source>`` tables as views of the
    parsed arrays; a second geometry naming the same source gets its own
    memory so editing one mesh leaves the other alone.
    """
    if key in doc.handed:
        return table.copy()
    doc.handed.add(key)
    return table


def _empty() -> PolyData:
    return PolyData(
        vertices=np.zeros((0, 3), dtype=np.float64),
        connectivity=np.zeros(0, dtype=np.int32),
        offsets=np.zeros(1, dtype=np.int32),
        element_types=np.zeros(0, dtype=np.uint8),
    )


def _mesh_at(st: _State, slot: int, bound: dict[str, int] | None) -> _Mesh:
    """Return the mesh of geometry slot ``slot``, reading it on first use."""
    mesh = st.meshes[slot]
    if mesh is None:
        mesh = _read_geometry(st.doc, st.geometries[slot], slot)
        st.meshes[slot] = mesh
    if any(sym is not None for sym, _ in mesh.symbols):
        if bound and (mesh.materials is None or not mesh.bound):
            mesh.bound = bound
            mesh.materials = _resolve_symbols(st, mesh, bound)
        elif bound and bound != mesh.bound:
            _warn(
                f"{st.doc.name!r}: geometry '{mesh.gid}' is bound to other materials "
                "by a later instance; the first binding is kept."
            )
    return mesh


def _resolve_deferred(st: _State, slot: int) -> None:
    """Resolve the symbols of a mesh no instance bound, now that none will."""
    mesh = st.meshes[slot]
    if (
        mesh is not None
        and mesh.materials is None
        and any(sym is not None for sym, _ in mesh.symbols)
    ):
        mesh.materials = _resolve_symbols(st, mesh, {})


def _resolve_symbols(st: _State, mesh: _Mesh, bound: dict[str, int]) -> np.ndarray:
    """Return the material index of every element, one lookup per distinct symbol."""
    index_of: dict[str | None, int] = {None: -1}
    missing: list[str] = []
    for sym, _ in mesh.symbols:
        if sym in index_of:
            continue
        idx = bound.get(sym)
        if idx is None:
            idx = st.material_index.get(sym)
        if idx is None:
            idx = -1
            missing.append(sym)
        index_of[sym] = idx
    per_run = np.array([index_of[sym] for sym, _ in mesh.symbols], dtype=np.int32)
    out = np.repeat(per_run, [n for _, n in mesh.symbols])
    if missing:
        _warn(
            f"{st.doc.name!r}: geometry '{mesh.gid}' names material symbol(s) "
            f"{missing} that no binding or material id resolves; those "
            "elements have material -1."
        )
    return out


def _finish_mesh(st: _State, mesh: _Mesh) -> PolyData:
    if mesh.materials is None:
        return mesh.poly
    return dataclasses.replace(
        mesh.poly, element_attrs={**mesh.poly.element_attrs, "material": mesh.materials}
    )


def _bind(st: _State, inst: ET.Element) -> dict[str, int]:
    """Return the ``symbol -> material index`` map an instance's bind_material gives."""
    out: dict[str, int] = {}
    tc = _child(_child(inst, "bind_material"), "technique_common")
    for im in _children(tc, "instance_material"):
        symbol = im.get("symbol")
        target = im.get("target", "")
        if symbol is None:
            continue
        if not target.startswith("#") or target[1:] not in st.material_index:
            _warn(
                f"{st.doc.name!r}: instance_material '{symbol}' targets "
                f"{target!r}, which is not a material of the document."
            )
            continue
        out[symbol] = st.material_index[target[1:]]
    return out


# -----------------------------------------------------------------------------
# nodes
# -----------------------------------------------------------------------------


def _rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return _IDENTITY.copy()
    x, y, z = axis / norm
    c = np.cos(np.radians(degrees))
    s = np.sin(np.radians(degrees))
    t = 1.0 - c
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
    return out


def _lookat(values: np.ndarray) -> np.ndarray:
    eye, target, up = values[:3], values[3:6], values[6:]
    z = eye - target
    zn = np.linalg.norm(z)
    z = z / zn if zn else np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    xn = np.linalg.norm(x)
    x = x / xn if xn else np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    out = np.eye(4, dtype=np.float64)
    out[:3, 0], out[:3, 1], out[:3, 2], out[:3, 3] = x, y, z, eye
    return out


def _matrix_of(kind: str, values: np.ndarray) -> np.ndarray:
    if kind == "matrix":
        return values.reshape(4, 4)
    if kind == "translate":
        out = np.eye(4, dtype=np.float64)
        out[:3, 3] = values
        return out
    if kind == "rotate":
        return _rotation(values[:3], float(values[3]))
    if kind == "scale":
        return np.diag([values[0], values[1], values[2], 1.0])
    return _lookat(values)


def _node_transforms(
    doc: _Doc, node: ET.Element, what: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrix = np.eye(4, dtype=np.float64)
    spelled: list[dict[str, Any]] = []
    for child in node:
        kind = _local(child.tag)
        if kind == "skew":
            _warn(f"{what} has a <skew>, which this reader ignores.")
            continue
        size = _KIND_SIZE.get(kind)
        if size is None:
            continue
        values = _floats(child.text, f"{what} <{kind}>", size)
        matrix = matrix @ _matrix_of(kind, values)
        spelled.append(
            {"kind": kind, "sid": child.get("sid"), "values": values.tolist()}
        )
    return matrix, spelled


def _controller_geometry(
    st: _State, ctrl: ET.Element, stack: tuple[str, ...]
) -> tuple[int, ET.Element | None]:
    """Return (geometry slot, skin element or None) a controller ends at."""
    cid = ctrl.get("id", "?")
    if cid in stack:
        raise CodecError(f"{st.doc.name!r}: controller '{cid}' contains itself.")
    skin = _child(ctrl, "skin")
    morph = _child(ctrl, "morph")
    body = skin if skin is not None else morph
    if body is None:
        raise CodecError(
            f"{st.doc.name!r}: controller '{cid}' has neither <skin> nor <morph>."
        )
    if morph is not None and skin is None:
        _warn(
            f"{st.doc.name!r}: controller '{cid}' is a morph; only its base geometry is read."
        )
    target = _ref(st.doc, body.get("source"), f"controller '{cid}'")
    if _local(target.tag) == "controller":
        slot, inner = _controller_geometry(st, target, (*stack, cid))
        return slot, skin if skin is not None else inner
    if _local(target.tag) != "geometry" or target.get("id") not in st.slot_of_gid:
        raise CodecError(
            f"{st.doc.name!r}: controller '{cid}' names {body.get('source')!r}, "
            "which is not a geometry."
        )
    return st.slot_of_gid[target.get("id", "")], skin


def _walk(st: _State, node: ET.Element, stack: tuple[str, ...]) -> int:
    doc = st.doc
    nid = node.get("id")
    what = f"{doc.name!r}: node '{nid or node.get('name') or '?'}'"
    if nid is not None and nid in stack:
        chain = " -> ".join((*stack, nid))
        raise CodecError(f"{doc.name!r}: node {chain} contains itself.")
    idx = len(st.nodes)
    st.nodes.append(None)
    inner = (*stack, nid) if nid is not None else stack
    if nid is not None:
        st.index_by_id.setdefault(nid, idx)
    sid = node.get("sid")
    if sid is not None:
        st.index_by_sid.setdefault(sid, []).append(idx)

    matrix, spelled = _node_transforms(doc, node, what)
    extras: dict[str, Any] = {}
    if nid is not None:
        extras["id"] = nid
    if sid is not None:
        extras["sid"] = sid
    if node.get("type") == "JOINT":
        extras["type"] = "JOINT"
    if spelled:
        extras["transforms"] = spelled

    mesh_idx: int | None = None
    children: list[int] = []
    for child in node:
        tag = _local(child.tag)
        if tag == "node":
            children.append(_walk(st, child, inner))
        elif tag == "instance_node":
            target = _ref(doc, child.get("url"), f"{what} instance_node")
            if _local(target.tag) != "node":
                raise CodecError(
                    f"{what} instances {child.get('url')!r}, which is not a node."
                )
            children.append(_walk(st, target, inner))
        elif tag == "instance_geometry":
            geo = _ref(doc, child.get("url"), f"{what} instance_geometry")
            slot = st.slot_of_gid.get(geo.get("id", ""))
            if slot is None or _local(geo.tag) != "geometry":
                raise CodecError(
                    f"{what} instances {child.get('url')!r}, which is not a geometry."
                )
            mesh = _mesh_at(st, slot, _bind(st, child))
            if mesh_idx is None:
                mesh_idx = slot
            else:
                children.append(
                    _synth_node(st, slot, st.geometries[slot].get("name", ""))
                )
        elif tag == "instance_controller":
            ctrl = _ref(doc, child.get("url"), f"{what} instance_controller")
            if _local(ctrl.tag) != "controller":
                raise CodecError(
                    f"{what} instances {child.get('url')!r}, which is not a controller."
                )
            slot, skin = _controller_geometry(st, ctrl, ())
            mesh = _mesh_at(st, slot, _bind(st, child))
            skin_extras: dict[str, Any] = {}
            if skin is not None:
                skeletons = [
                    (sk.text or "").strip().removeprefix("#")
                    for sk in _children(child, "skeleton")
                ]
                skin_extras["skin"] = _read_skin(st, ctrl, skin, slot, mesh, skeletons)
            if mesh_idx is None:
                mesh_idx = slot
                extras.update(skin_extras)
            else:
                children.append(
                    _synth_node(st, slot, ctrl.get("name", ""), skin_extras)
                )
        elif tag in ("instance_camera", "instance_light"):
            url = child.get("url", "")
            extras[tag[len("instance_") :]] = url[1:] if url.startswith("#") else url

    st.nodes[idx] = SceneNode(
        name=node.get("name", ""),
        mesh=mesh_idx,
        children=tuple(children),
        matrix=matrix,
        extras=extras,
    )
    return idx


def _synth_node(
    st: _State, slot: int, name: str, extras: dict[str, Any] | None = None
) -> int:
    st.nodes.append(SceneNode(name=name, mesh=slot, extras=extras or {}))
    return len(st.nodes) - 1


def _read_scenes(st: _State) -> tuple[tuple[tuple[int, ...], ...], int, str]:
    doc = st.doc
    visual = _library(doc.root, "library_visual_scenes", "visual_scene")
    scenes: list[tuple[int, ...]] = []
    for vs in visual:
        roots: list[int] = []
        for child in vs:
            tag = _local(child.tag)
            if tag == "node":
                roots.append(_walk(st, child, ()))
            elif tag == "instance_node":
                what = f"{doc.name!r}: visual_scene instance_node"
                target = _ref(doc, child.get("url"), what)
                if _local(target.tag) != "node":
                    raise CodecError(
                        f"{what} instances {child.get('url')!r}, which is not a node."
                    )
                roots.append(_walk(st, target, ()))
        scenes.append(tuple(roots))
    active = 0
    inst = _child(_child(doc.root, "scene"), "instance_visual_scene")
    if inst is not None and visual:
        target = _ref(doc, inst.get("url"), f"{doc.name!r}: <scene>")
        for i, vs in enumerate(visual):
            if vs is target:
                active = i
                break
        else:
            raise CodecError(
                f"{doc.name!r}: <scene> instances {inst.get('url')!r}, which is not "
                "a visual_scene."
            )
    name = visual[active].get("name", "") if visual else ""
    return tuple(scenes), active, name


# -----------------------------------------------------------------------------
# skins
# -----------------------------------------------------------------------------


def _read_skin(
    st: _State,
    ctrl: ET.Element,
    skin: ET.Element,
    slot: int,
    mesh: _Mesh,
    skeletons: list[str],
) -> int:
    doc = st.doc
    cid = ctrl.get("id", "?")
    known = st.skin_of_ctrl.get(cid)
    if known is not None:
        return known
    what = f"{doc.name!r}: skin '{cid}'"
    bsm_elem = _child(skin, "bind_shape_matrix")
    bind_shape = (
        _floats(bsm_elem.text, f"{what} bind_shape_matrix", 16).reshape(4, 4)
        if bsm_elem is not None
        else np.eye(4)
    )

    joints_elem = _child(skin, "joints")
    names: list[str] = []
    names_src: ET.Element | None = None
    ibm = np.zeros((0, 4, 4))
    for inp in _children(joints_elem, "input"):
        semantic = inp.get("semantic")
        src = _ref(doc, inp.get("source"), f"{what} joints {semantic} input")
        if semantic == "JOINT":
            names = _name_source(doc, src, f"{what} JOINT")
            names_src = src
        elif semantic == "INV_BIND_MATRIX":
            table = _float_source(doc, src, f"{what} INV_BIND_MATRIX")
            if table.shape[1] != 16:
                raise CodecError(
                    f"{what}: INV_BIND_MATRIX has stride {table.shape[1]}, not 16."
                )
            ibm = table.reshape(-1, 4, 4)
    if len(ibm) and len(ibm) != len(names):
        raise CodecError(
            f"{what} names {len(names)} joints but holds {len(ibm)} inverse bind matrices."
        )
    if not len(ibm):
        ibm = np.tile(np.eye(4), (len(names), 1, 1))

    vw = _child(skin, "vertex_weights")
    n_pos = mesh.n_positions
    joints_pp = np.zeros((n_pos, 4), dtype=np.int32)
    weights_pp = np.zeros((n_pos, 4), dtype=np.float64)
    if vw is not None:
        count = _int(vw.get("count"), f"{what} vertex_weights count")
        inputs = _children(vw, "input")
        offsets = [
            _int(i.get("offset"), f"{what} vertex_weights offset", default=0)
            for i in inputs
        ]
        stride = max(offsets, default=-1) + 1
        joint_col = weight_col = None
        weight_values = np.zeros(0)
        joint_names = names
        same_source = True
        for inp, off in zip(inputs, offsets, strict=True):
            semantic = inp.get("semantic")
            src = _ref(
                doc, inp.get("source"), f"{what} vertex_weights {semantic} input"
            )
            if semantic == "JOINT":
                joint_col = off
                joint_names = _name_source(doc, src, f"{what} JOINT")
                same_source = src is names_src
            elif semantic == "WEIGHT":
                weight_col = off
                weight_values = _float_source(doc, src, f"{what} WEIGHT")[:, 0]
        if joint_col is None or weight_col is None:
            raise CodecError(f"{what}: vertex_weights needs JOINT and WEIGHT inputs.")
        vc = _child(vw, "vcount")
        vcount = _ints(vc.text if vc is not None else "", f"{what} <vcount>")
        if len(vcount) != count:
            raise CodecError(
                f"{what}: vertex_weights declares count={count} but <vcount> holds {len(vcount)}."
            )
        if len(vcount) and vcount.min() < 0:
            raise CodecError(f"{what}: <vcount> has an entry below 0.")
        v_elem = _child(vw, "v")
        v = _ints(v_elem.text if v_elem is not None else "", f"{what} <v>")
        need = int(vcount.sum()) * stride
        if len(v) != need:
            raise CodecError(
                f"{what}: vertex_weights <vcount> sums to {need} indices but <v> holds {len(v)}."
            )
        if count > n_pos:
            raise CodecError(
                f"{what}: vertex_weights covers {count} vertices but the geometry has {n_pos}."
            )
        if count < n_pos:
            _warn(
                f"{what}: vertex_weights covers {count} of the geometry's {n_pos} "
                "vertices; the rest have no influence."
            )
        pairs = v.reshape(-1, stride)
        j = pairs[:, joint_col]
        w = pairs[:, weight_col]
        if len(w) and (w.min() < 0 or w.max() >= len(weight_values)):
            raise CodecError(f"{what}: a <v> weight index is past the WEIGHT source.")
        if len(j) and j.max() >= len(joint_names):
            raise CodecError(f"{what}: a <v> joint index is past the JOINT source.")
        if same_source:
            slots = j
        else:
            slot_of_name: dict[str, int] = {}
            for i, n in enumerate(names):
                slot_of_name.setdefault(n, i)
            of_joint = np.array([slot_of_name.get(n, -1) for n in joint_names] + [-1])
            slots = of_joint[np.where(j >= 0, j, len(joint_names))]
        _pack_influences(what, vcount, slots, weight_values[w], joints_pp, weights_pp)

    if mesh.skinned:
        _warn(
            f"{what}: geometry '{mesh.gid}' is skinned twice; the first skin's weights are kept."
        )
    else:
        mesh.poly = dataclasses.replace(
            mesh.poly,
            vertex_attrs={
                **mesh.poly.vertex_attrs,
                "joints": joints_pp[mesh.positions_of],
                "weights": weights_pp[mesh.positions_of],
            },
        )
        mesh.skinned = True

    skin_idx = len(st.skins)
    st.skins.append(
        {
            "name": ctrl.get("name") or cid,
            "mesh": slot,
            "joints": [-1] * len(names),
            "inverse_bind_matrices": ibm,
            "bind_shape_matrix": bind_shape,
        }
    )
    st.skin_of_ctrl[cid] = skin_idx
    st.pending_joints.append((skin_idx, names, skeletons))
    return skin_idx


def _pack_influences(
    what: str,
    vcount: np.ndarray,
    slots: np.ndarray,
    w: np.ndarray,
    joints_pp: np.ndarray,
    weights_pp: np.ndarray,
) -> None:
    """Scatter the ``<v>`` pairs into four influence slots per vertex.

    A negative slot (the bind shape, or a joint name the skin's ``<joints>``
    do not list) is dropped. A vertex with more than four influences keeps
    the four heaviest, renormalised; every other vertex keeps its file
    order.
    """
    n = len(vcount)
    vid = np.repeat(np.arange(n), vcount)
    keep = slots >= 0
    vid, slots, w = vid[keep], slots[keep], w[keep]
    per_vertex = np.bincount(vid, minlength=n)
    crowded = per_vertex[vid] > 4
    order = np.lexsort((np.arange(len(vid)), np.where(crowded, -w, 0.0), vid))
    vid, slots, w, crowded = vid[order], slots[order], w[order], crowded[order]
    rank = np.arange(len(vid)) - (np.cumsum(per_vertex) - per_vertex)[vid]
    top = rank < 4
    if crowded.any():
        total = np.bincount(vid[top], weights=w[top], minlength=n)
        scale = np.where(total > 0, 1.0 / np.where(total > 0, total, 1.0), 1.0)
        w = np.where(crowded, w * scale[vid], w)
        _warn(
            f"{what} gives some vertices more than four influences; the four "
            "heaviest are kept and renormalised."
        )
    joints_pp[vid[top], rank[top]] = slots[top]
    weights_pp[vid[top], rank[top]] = w[top]


def _subtree(st: _State, roots: list[str]) -> set[int] | None:
    """Return the node indices under the ``<skeleton>`` roots, None when none resolve."""
    under: set[int] = set()
    stack = [st.index_by_id[r] for r in roots if r in st.index_by_id]
    if not stack:
        return None
    while stack:
        idx = stack.pop()
        if idx in under:
            continue
        under.add(idx)
        node = st.nodes[idx]
        if node is not None:
            stack.extend(node.children)
    return under


def _resolve_joints(st: _State) -> None:
    for skin_idx, names, skeletons in st.pending_joints:
        under = _subtree(st, skeletons)
        resolved: list[int] = []
        missing: list[str] = []
        for n in names:
            by_sid = st.index_by_sid.get(n, [])
            by_id = st.index_by_id.get(n)
            scoped = [i for i in by_sid if i in under] if under is not None else by_sid
            if scoped:
                resolved.append(scoped[0])
            elif by_id is not None and under is not None and by_id in under:
                resolved.append(by_id)
            elif by_sid:
                resolved.append(by_sid[0])
            elif by_id is not None:
                resolved.append(by_id)
            else:
                resolved.append(-1)
                missing.append(n)
        st.skins[skin_idx]["joints"] = resolved
        if missing:
            _warn(
                f"{st.doc.name!r}: skin '{st.skins[skin_idx]['name']}' names joint(s) "
                f"{missing} that no node's sid or id matches; they are -1."
            )


# -----------------------------------------------------------------------------
# animations
# -----------------------------------------------------------------------------


def _sampler(
    st: _State, elem: ET.Element, cache: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    doc = st.doc
    known = cache.get(id(elem))
    if known is not None:
        return known
    what = f"{doc.name!r}: sampler '{elem.get('id', '?')}'"
    out: dict[str, Any] = {"interpolation": "LINEAR"}
    times = values = None
    for inp in _children(elem, "input"):
        semantic = inp.get("semantic")
        src = _ref(doc, inp.get("source"), f"{what} {semantic} input")
        if semantic == "INPUT":
            times = _float_source(doc, src, f"{what} INPUT")[:, 0]
        elif semantic == "OUTPUT":
            values = _float_source(doc, src, f"{what} OUTPUT")
        elif semantic == "INTERPOLATION":
            names = _name_source(doc, src, f"{what} INTERPOLATION")
            if names:
                if len(set(names)) > 1:
                    _warn(
                        f"{what} mixes interpolations; {names[0]} is taken for all keys."
                    )
                out["interpolation"] = names[0]
        elif semantic in ("IN_TANGENT", "OUT_TANGENT"):
            out[semantic.lower() + "s"] = _float_source(doc, src, f"{what} {semantic}")
    if times is None or values is None:
        raise CodecError(f"{what} needs INPUT and OUTPUT.")
    if len(times) != len(values):
        raise CodecError(
            f"{what} has {len(times)} keys but {len(values)} output values."
        )
    out["times"] = times
    out["values"] = values[:, 0] if values.shape[1] == 1 else values
    cache[id(elem)] = out
    return out


def _target(st: _State, target: str) -> dict[str, Any] | None:
    parts = target.split("/")
    if len(parts) < 2:
        return None
    node_idx = st.index_by_id.get(parts[0])
    if node_idx is None:
        return None
    for part in parts[1:-1]:
        node = st.nodes[node_idx]
        node_idx = next(
            (c for c in node.children if st.nodes[c].extras.get("sid") == part),
            None,
        )
        if node_idx is None:
            return None
    last = parts[-1]
    member: str | None = None
    for i, ch in enumerate(last):
        if ch in ".(":
            member = last[i + 1 :] if ch == "." else last[i:]
            last = last[:i]
            break
    for t in st.nodes[node_idx].extras.get("transforms", ()):
        if t["sid"] == last:
            return {
                "node": node_idx,
                "path": _PATH_OF_KIND[t["kind"]],
                "sid": last,
                "member": member,
            }
    return None


def _read_animations(st: _State) -> list[dict[str, Any]]:
    doc = st.doc
    out: list[dict[str, Any]] = []
    cache: dict[int, dict[str, Any]] = {}
    for anim in _library(doc.root, "library_animations", "animation"):
        samplers: list[dict[str, Any]] = []
        slot_of: dict[int, int] = {}
        channels: list[dict[str, Any]] = []
        for channel in (e for e in anim.iter() if _local(e.tag) == "channel"):
            target_text = channel.get("target", "")
            target = _target(st, target_text)
            if target is None:
                _warn(
                    f"{doc.name!r}: animation '{anim.get('id', '?')}' targets "
                    f"{target_text!r}, which reaches no node transform; the channel is dropped."
                )
                continue
            elem = _ref(doc, channel.get("source"), f"{doc.name!r}: a channel")
            if _local(elem.tag) != "sampler":
                raise CodecError(
                    f"{doc.name!r}: channel {target_text!r} names {channel.get('source')!r}, "
                    "which is not a sampler."
                )
            sampler = _sampler(st, elem, cache)
            slot = slot_of.get(id(elem))
            if slot is None:
                slot = slot_of[id(elem)] = len(samplers)
                samplers.append(sampler)
            channels.append({"sampler": slot, "target": target})
        if not channels:
            continue
        out.append(
            {
                "name": anim.get("name") or anim.get("id", ""),
                "channels": channels,
                "samplers": samplers,
            }
        )
    return out


# =============================================================================
# write
# =============================================================================


def write_scene(
    scene: SceneData,
    path: Source,
    *,
    up_axis: str | None = None,
    unit: dict[str, Any] | None = None,
    **opts: Any,
) -> None:
    """Write a SceneData as a COLLADA 1.4.1 document.

    Parameters
    ----------
    scene
        Scene to serialize. Every mesh becomes a ``<geometry>`` named by its
        ``global_attrs["mesh_name"]``, every node a ``<node>`` spelled with
        the transform elements it was read with when ``extras["transforms"]``
        still composes to its matrix, else one ``<matrix>``; materials
        become phong effects, skins controllers and animations channels
        targeting those transform elements. COLLADA has no alpha mask: a
        material's alpha goes out as an ``A_ONE`` transparency when it is
        below one or ``alpha_mode`` is not ``OPAQUE``, and ``alpha_mode``
        and ``alpha_cutoff`` themselves are not written.
    path
        Output file path or open binary file object.
    up_axis
        ``X_UP``, ``Y_UP`` or ``Z_UP``; overrides ``global_attrs["asset"]``.
    unit
        ``{"name": ..., "meter": ...}``; overrides ``global_attrs["asset"]``.
    **opts
        Accepted for interface symmetry and ignored.

    Warns
    -----
    UserWarning
        For elements COLLADA cannot hold (points, volume cells; a mesh with
        nothing else keeps its positions as a ``<mesh>`` without primitives),
        vertex attributes it has no input for, a ``texcoords`` or ``colors``
        set suffix that is not a free number (it is written under the lowest
        free one and reads back under it), a node no scene reaches (it
        is not written, and a skin joint or animation channel naming it is
        told so), a transform ``sid`` that is not an XML name, an animation
        channel whose target the written node lacks, or a skin its mesh has
        no weights for.

    Raises
    ------
    CodecError
        When the node graph has a cycle, a name holds a character XML
        cannot carry, ``up_axis`` is not one of the three, a skin's joint
        index is past its joint list, its inverse bind matrices do not
        number its joints, or an animation sampler's ``interpolation`` is
        not one of the six the specification names.
    """
    name = source_name(path)
    asset = dict(scene.global_attrs.get("asset", {}))
    axis = up_axis or asset.get("up_axis", "Y_UP")
    if axis not in _UP_AXES:
        raise CodecError(
            f"{name!r}: up_axis must be one of {', '.join(_UP_AXES)}, not {axis!r}."
        )
    unit = unit or asset.get("unit") or {"name": "meter", "meter": 1.0}

    ids = _Ids(scene)
    parts: list[str] = [
        '<?xml version="1.0" encoding="utf-8"?>\n',
        f'<COLLADA xmlns="{_NS}" version="1.4.1">\n',
        "  <asset>\n",
        f"    <contributor><authoring_tool>polyxios {__version__}</authoring_tool></contributor>\n",
        f"    <created>{_esc(str(asset.get('created', _EPOCH)), 'asset created')}</created>\n",
        f"    <modified>{_esc(str(asset.get('modified', _EPOCH)), 'asset modified')}</modified>\n",
        f'    <unit name={_attr(str(unit.get("name", "meter")), "unit name")} meter="{float(unit.get("meter", 1.0))!r}"/>\n',
        f"    <up_axis>{axis}</up_axis>\n",
        "  </asset>\n",
    ]
    if scene.images:
        parts.append("  <library_images>\n")
        for i, img in enumerate(scene.images):
            parts.append(_image_xml(ids.image(i), img))
        parts.append("  </library_images>\n")
    if scene.materials:
        parts.append("  <library_effects>\n")
        for i, mat in enumerate(scene.materials):
            parts.append(_effect_xml(scene, ids, i, mat, name))
        parts.append("  </library_effects>\n  <library_materials>\n")
        for i, mat in enumerate(scene.materials):
            parts.append(
                f'    <material id="{ids.material(i)}"{_name_attr(mat.name)}>'
                f'<instance_effect url="#{ids.effect(i)}"/></material>\n'
            )
        parts.append("  </library_materials>\n")

    symbols_of: list[list[int]] = []
    parts.append("  <library_geometries>\n")
    for i, mesh in enumerate(scene.meshes):
        xml, used = _geometry_xml(mesh, ids, i, len(scene.materials), name)
        parts.append(xml)
        symbols_of.append(used)
    parts.append("  </library_geometries>\n")

    roots_per_scene = list(scene.scenes) if scene.scenes else [_roots(scene)]
    written = _reachable(scene, roots_per_scene)
    if len(written) < len(scene.nodes):
        _warn(
            f"{name!r}: {len(scene.nodes) - len(written)} node(s) are in no scene "
            "and are not written."
        )

    controllers = _controllers_xml(scene, ids, written, name)
    if controllers:
        parts.append(
            "  <library_controllers>\n"
            + "".join(xml for xml, _, _ in controllers.values())
            + "  </library_controllers>\n"
        )

    animations = _animations_xml(scene, ids, written, name)
    if animations:
        parts.append(
            "  <library_animations>\n" + animations + "  </library_animations>\n"
        )

    parts.append("  <library_visual_scenes>\n")
    for s, roots in enumerate(roots_per_scene):
        label = _name_attr(scene.name) if s == scene.active_scene else ""
        parts.append(f'    <visual_scene id="{ids.scene(s)}"{label}>\n')
        parts.extend(
            _node_xml(scene, ids, r, symbols_of, controllers, (), 3, name)
            for r in roots
        )
        parts.append("    </visual_scene>\n")
    parts.append("  </library_visual_scenes>\n")
    active = scene.active_scene if 0 <= scene.active_scene < len(roots_per_scene) else 0
    parts.append(
        f'  <scene><instance_visual_scene url="#{ids.scene(active)}"/></scene>\n</COLLADA>\n'
    )
    write_text(path, "".join(parts))


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a PolyData as a one-node COLLADA scene.

    Parameters
    ----------
    poly
        Mesh to serialize. ``element_attrs["material"]`` values become
        materials named ``material_<value>``; points and volume cells are
        skipped with a warning.
    path
        Output file path or open binary file object.
    **opts
        Passed to :func:`write_scene` (``up_axis``, ``unit``).

    Raises
    ------
    CodecError
        Whatever :func:`write_scene` refuses.
    """
    materials: tuple[SceneMaterial, ...] = ()
    column = poly.element_attrs.get("material")
    if column is not None:
        column = np.asarray(column).ravel()
        used = np.unique(column[column >= 0])
        materials = tuple(
            SceneMaterial(
                name=f"material_{int(m)}", metallic=0.0, extras={"shading": "phong"}
            )
            for m in used
        )
        new_column = np.where(column >= 0, np.searchsorted(used, column), -1).astype(
            np.int32
        )
        poly = dataclasses.replace(
            poly, element_attrs={**poly.element_attrs, "material": new_column}
        )
    global_attrs = {}
    if "asset" in poly.global_attrs:
        global_attrs["asset"] = poly.global_attrs["asset"]
    scene = SceneData(
        meshes=(poly,),
        nodes=(SceneNode(mesh=0),),
        materials=materials,
        scenes=((0,),),
        global_attrs=global_attrs,
    )
    write_scene(scene, path, **opts)


# -----------------------------------------------------------------------------
# write helpers
# -----------------------------------------------------------------------------


class _Ids:
    """The XML ids a write uses, given ones kept where they are valid and unique.

    A generated id also serves as the stem of ``<stem>-positions``,
    ``<stem>-array`` and the like, so a stem is not handed out while a
    given node id begins with it and a dash.
    """

    def __init__(self, scene: SceneData) -> None:
        self.taken: set[str] = set()
        self.stems: set[str] = set()
        self.node_ids: list[str] = []
        given = [
            n.extras.get("id")
            if isinstance(n.extras.get("id"), str) and _NCNAME.match(n.extras["id"])
            else None
            for n in scene.nodes
        ]
        for i, g in enumerate(given):
            if g is not None and g not in self.taken:
                self.taken.add(g)
                head = g
                while "-" in head:
                    head = head.rsplit("-", 1)[0]
                    self.stems.add(head)
            elif g is not None:
                given[i] = None
        for i, g in enumerate(given):
            self.node_ids.append(g if g is not None else self._fresh(f"node{i}"))
        self.node_sids: list[str] = [
            n.extras["sid"]
            if isinstance(n.extras.get("sid"), str) and _NCNAME.match(n.extras["sid"])
            else self.node_ids[i]
            for i, n in enumerate(scene.nodes)
        ]
        self._cache: dict[tuple[str, int], str] = {}

    def _fresh(self, base: str) -> str:
        candidate = base
        k = 0
        while candidate in self.taken or candidate in self.stems:
            k += 1
            candidate = f"{base}_{k}"
        self.taken.add(candidate)
        return candidate

    def _of(self, prefix: str, i: int) -> str:
        key = (prefix, i)
        if key not in self._cache:
            self._cache[key] = self._fresh(f"{prefix}{i}")
        return self._cache[key]

    def image(self, i: int) -> str:
        return self._of("image", i)

    def effect(self, i: int) -> str:
        return self._of("effect", i)

    def material(self, i: int) -> str:
        return self._of("material", i)

    def geometry(self, i: int) -> str:
        return self._of("geometry", i)

    def scene(self, i: int) -> str:
        return self._of("scene", i)

    def skin(self, i: int) -> str:
        return self._of("skin", i)

    def animation(self, i: int) -> str:
        return self._of("animation", i)

    def node(self, i: int) -> str:
        return self.node_ids[i]

    def sid(self, i: int) -> str:
        return self.node_sids[i]


def _transform_sid(value: Any) -> str | None:
    return value if isinstance(value, str) and _NCNAME.match(value) else None


def _esc(text: str, what: str) -> str:
    if _XML_FORBIDDEN.search(text):
        raise CodecError(f"{what} holds a character XML cannot carry: {text!r}.")
    return escape(text)


def _attr(text: str, what: str) -> str:
    if _XML_FORBIDDEN.search(text):
        raise CodecError(f"{what} holds a character XML cannot carry: {text!r}.")
    return '"' + escape(text, _ATTR_ENTITIES) + '"'


def _name_attr(name: str) -> str:
    return f" name={_attr(name, 'a name')}" if name else ""


def _nums(values: np.ndarray) -> str:
    flat = np.asarray(values).ravel()
    words = list(map(str, flat.tolist()))
    if flat.dtype.kind == "f":
        for i in np.flatnonzero(~np.isfinite(flat)).tolist():
            v = flat[i]
            words[i] = "NaN" if v != v else ("INF" if v > 0 else "-INF")
    return " ".join(words)


def _source_xml(
    sid: str,
    table: np.ndarray,
    params: tuple[str, ...],
    *,
    indent: int = 8,
    kind: str = "float",
) -> str:
    table = np.asarray(table)
    stride = table.shape[1] if table.ndim == 2 else 1
    count = table.shape[0] if table.ndim else 0
    pad = " " * indent
    plist = "".join(
        f'<param name="{p}" type="{kind if p != "TRANSFORM" else "float4x4"}"/>'
        for p in params
    )
    return (
        f'{pad}<source id="{sid}">\n'
        f'{pad}  <float_array id="{sid}-array" count="{count * stride}">{_nums(table)}</float_array>\n'
        f'{pad}  <technique_common><accessor source="#{sid}-array" count="{count}" stride="{stride}">{plist}</accessor></technique_common>\n'
        f"{pad}</source>\n"
    )


def _name_source_xml(sid: str, names: list[str], param: str, *, indent: int = 8) -> str:
    pad = " " * indent
    return (
        f'{pad}<source id="{sid}">\n'
        f'{pad}  <Name_array id="{sid}-array" count="{len(names)}">{" ".join(names)}</Name_array>\n'
        f'{pad}  <technique_common><accessor source="#{sid}-array" count="{len(names)}" stride="1"><param name="{param}" type="name"/></accessor></technique_common>\n'
        f"{pad}</source>\n"
    )


def _image_xml(iid: str, img: SceneImage) -> str:
    if img.data is not None:
        media = img.media_type or "application/octet-stream"
        uri = f"data:{media};base64,{base64.b64encode(img.data).decode('ascii')}"
    else:
        uri = img.uri or ""
    return (
        f'    <image id="{iid}"{_name_attr(img.name)}>'
        f"<init_from>{_esc(uri, 'an image uri')}</init_from></image>\n"
    )


def _color_xml(tag: str, rgba: tuple[float, ...]) -> str:
    values = tuple(rgba) + (1.0,) * (4 - len(rgba))
    return f"          <{tag}><color>{_nums(np.array(values[:4]))}</color></{tag}>\n"


def _effect_xml(
    scene: SceneData, ids: _Ids, i: int, mat: SceneMaterial, name: str
) -> str:
    eid = ids.effect(i)
    samplers: dict[str, str] = {}
    params: list[str] = []
    for key in _MATERIAL_TEXTURES:
        t = getattr(mat, key)
        if t is None:
            continue
        if not (0 <= t < len(scene.textures)) or not (
            0 <= scene.textures[t].image < len(scene.images)
        ):
            _warn(
                f"{name!r}: material {i} names texture {t}, which reaches no image; it is dropped."
            )
            continue
        tex = scene.textures[t]
        surface = f"{eid}-surface{t}"
        sampler = f"{eid}-sampler{t}"
        settings = ""
        if tex.wrap_s != 10497:
            settings += f"<wrap_s>{_GL_TO_WRAP.get(tex.wrap_s, 'WRAP')}</wrap_s>"
        if tex.wrap_t != 10497:
            settings += f"<wrap_t>{_GL_TO_WRAP.get(tex.wrap_t, 'WRAP')}</wrap_t>"
        if tex.min_filter in _GL_TO_FILTER:
            settings += f"<minfilter>{_GL_TO_FILTER[tex.min_filter]}</minfilter>"
        if tex.mag_filter in _GL_TO_FILTER:
            settings += f"<magfilter>{_GL_TO_FILTER[tex.mag_filter]}</magfilter>"
        if sampler not in samplers.values():
            params.append(
                f'        <newparam sid="{surface}"><surface type="2D"><init_from>{ids.image(tex.image)}</init_from></surface></newparam>\n'
                f'        <newparam sid="{sampler}"><sampler2D><source>{surface}</source>{settings}</sampler2D></newparam>\n'
            )
        samplers[key] = sampler

    shading = mat.extras.get("shading", "phong")
    if shading not in _SHADINGS:
        shading = "phong"
    body: list[str] = []

    def channel(tag: str, key: str | None, rgba: tuple[float, ...] | None) -> None:
        sampler = samplers.get(key) if key else None
        if sampler is not None:
            body.append(
                f'          <{tag}><texture texture="{sampler}" texcoord="UVMap"/></{tag}>\n'
            )
        elif rgba is not None:
            body.append(_color_xml(tag, rgba))

    channel("emission", "emissive_texture", tuple(mat.emissive))
    if shading != "constant":
        ambient = mat.extras.get("ambient")
        if ambient is not None:
            channel("ambient", None, tuple(ambient))
        channel(
            "diffuse",
            "base_color_texture",
            (mat.base_color[0], mat.base_color[1], mat.base_color[2], 1.0),
        )
    if shading in ("phong", "blinn"):
        specular = mat.extras.get("specular")
        if specular is not None:
            channel("specular", None, tuple(specular))
        shininess = mat.extras.get("shininess")
        if shininess is not None:
            body.append(
                f"          <shininess><float>{float(shininess)!r}</float></shininess>\n"
            )
    reflective = mat.extras.get("reflective")
    if reflective is not None:
        channel("reflective", None, tuple(reflective))
    reflectivity = mat.extras.get("reflectivity")
    if reflectivity is not None:
        body.append(
            f"          <reflectivity><float>{float(reflectivity)!r}</float></reflectivity>\n"
        )
    alpha = float(mat.base_color[3])
    if alpha < 1.0 or mat.alpha_mode != "OPAQUE":
        body.append(
            f'          <transparent opaque="A_ONE"><color>1 1 1 {alpha!r}</color></transparent>\n'
            "          <transparency><float>1.0</float></transparency>\n"
        )
    ior = mat.extras.get("index_of_refraction")
    if ior is not None:
        body.append(
            f"          <index_of_refraction><float>{float(ior)!r}</float></index_of_refraction>\n"
        )
    bump = samplers.get("normal_texture")
    extra = ""
    if bump is not None:
        extra = (
            '        <extra><technique profile="FCOLLADA"><bump>'
            f'<texture texture="{bump}" texcoord="UVMap"/></bump></technique></extra>\n'
        )
    double = (
        '      <extra><technique profile="GOOGLEEARTH"><double_sided>1</double_sided></technique></extra>\n'
        if mat.double_sided
        else ""
    )
    return (
        f'    <effect id="{eid}">\n      <profile_COMMON>\n'
        + "".join(params)
        + f'        <technique sid="common">\n        <{shading}>\n'
        + "".join(body)
        + f"        </{shading}>\n"
        + extra
        + "        </technique>\n"
        + double
        + "      </profile_COMMON>\n    </effect>\n"
    )


_ATTR_PARAMS = {
    "normals": (3, ("X", "Y", "Z"), "NORMAL"),
    "colors": (None, ("R", "G", "B", "A"), "COLOR"),
    "texcoords": (None, ("S", "T", "P"), "TEXCOORD"),
}


def _geometry_xml(
    mesh: PolyData, ids: _Ids, i: int, n_materials: int, name: str
) -> tuple[str, list[int]]:
    gid = ids.geometry(i)
    n = len(mesh.vertices)
    label = mesh.global_attrs.get("mesh_name")
    parts = [
        f'    <geometry id="{gid}"{_name_attr(label) if isinstance(label, str) else ""}>\n      <mesh>\n'
    ]
    parts.append(
        _source_xml(
            f"{gid}-positions",
            np.asarray(mesh.vertices, dtype=np.float64),
            ("X", "Y", "Z"),
        )
    )

    inputs: list[str] = []
    dropped: list[str] = []
    renamed: list[str] = []
    # Set numbers spelled by a key are reserved up front, so a key whose
    # suffix is not a free number takes one no key spells.
    used_sets: dict[str, set[int]] = {}
    for base in ("texcoords", "colors"):
        suffixes = [_set_of(k, base) for k in mesh.vertex_attrs]
        used_sets[base] = {int(s) for s in suffixes if s.isdigit()}
    given_sets = {base: set(taken) for base, taken in used_sets.items()}
    for key, table in mesh.vertex_attrs.items():
        if key in ("joints", "weights"):
            continue
        base = key
        set_no = 0
        for prefix in ("texcoords", "colors"):
            suffix = _set_of(key, prefix)
            if not suffix:
                continue
            base = prefix
            if suffix.isdigit() and int(suffix) in given_sets[prefix]:
                set_no = int(suffix)
                given_sets[prefix].discard(set_no)
            else:
                set_no = 0
                while set_no in used_sets[prefix]:
                    set_no += 1
                used_sets[prefix].add(set_no)
                renamed.append(f"{key} as set {set_no}")
        spec = _ATTR_PARAMS.get(base)
        arr = np.asarray(table)
        if (
            spec is None
            or arr.ndim != 2
            or len(arr) != n
            or arr.shape[1] > len(spec[1])
            or arr.shape[1] < 2
            or (spec[0] is not None and arr.shape[1] != spec[0])
            or not np.issubdtype(arr.dtype, np.number)
        ):
            dropped.append(key)
            continue
        if base == "colors" and arr.shape[1] == 2:
            dropped.append(key)
            continue
        sid = f"{gid}-{base}" + (f"_{set_no}" if set_no else "")
        parts.append(_source_xml(sid, arr.astype(np.float64), spec[1][: arr.shape[1]]))
        inputs.append(
            f'<input semantic="{spec[2]}" source="#{sid}" offset="0" set="{set_no}"/>'
        )
    if renamed:
        _warn(
            f"{name!r}: vertex attribute(s) {renamed} of mesh {i} have a set that "
            "is not a free number (COLLADA's set is an unsigned integer); each "
            "reads back under the number it is written with."
        )
    if dropped:
        _warn(
            f"{name!r}: vertex attribute(s) {dropped} of mesh {i} have no COLLADA "
            "input (only normals, texcoords and colors of 2-4 columns do); they are dropped."
        )
    parts.append(
        f'        <vertices id="{gid}-vertices"><input semantic="POSITION" source="#{gid}-positions"/></vertices>\n'
    )

    material = mesh.element_attrs.get("material")
    types = np.asarray(mesh.element_types)
    if material is not None:
        mat_of = np.asarray(material).ravel().astype(np.int64)
        if len(mat_of) and mat_of.max() >= n_materials:
            raise CodecError(
                f"{name!r}: mesh {i} names material {int(mat_of.max())}, but the "
                f"scene holds {n_materials} material(s)."
            )
        mat_of = np.maximum(mat_of, -1)
    else:
        mat_of = np.full(len(types), -1, dtype=np.int64)
    block_of = np.array([_BLOCK_INDEX.get(code, -1) for code in range(256)])[types]
    # A mesh mixing triangles with quads or polygons writes them all in one
    # polylist, so those elements keep their order on the way back.
    if np.isin(types, [_QUAD, _POLYGON]).any():
        block_of[types == _TRI] = _BLOCKS.index("polylist")
    writable = block_of >= 0
    skipped = [
        ELEMENT_TYPES_INV.get(int(code), str(code))
        for code in np.unique(types[~writable])
    ]
    if skipped:
        _warn(
            f"{name!r}: element type(s) {skipped} of mesh {i} have no COLLADA "
            "primitive; they are skipped"
            + (", leaving the positions alone." if not writable.any() else ".")
        )

    used: list[int] = []
    conn = np.asarray(mesh.connectivity)
    offs = np.asarray(mesh.offsets)
    input_xml = (
        f'<input semantic="VERTEX" source="#{gid}-vertices" offset="0"/>'
        + "".join(inputs)
    )
    keys = np.where(writable, block_of * (n_materials + 1) + mat_of + 1, -1)
    _, first = np.unique(keys, return_index=True)
    for start in np.sort(first).tolist():
        if keys[start] < 0:
            continue
        elems = np.flatnonzero(keys == keys[start])
        block = _BLOCKS[int(block_of[start])]
        m = int(mat_of[start])
        mat_attr = ""
        if m >= 0:
            mat_attr = f' material="{ids.material(m)}"'
            if m not in used:
                used.append(m)
        sizes = offs[elems + 1] - offs[elems]
        parts.append(f'        <{block} count="{len(elems)}"{mat_attr}>{input_xml}')
        if block in ("linestrips", "tristrips"):
            parts.append(
                "".join(
                    f"<p>{_nums(conn[offs[e] : offs[e + 1]])}</p>"
                    for e in elems.tolist()
                )
            )
        else:
            gathered = conn[_gather(offs[elems], sizes)]
            if block == "polylist":
                parts.append(f"<vcount>{_nums(sizes)}</vcount>")
            parts.append(f"<p>{_nums(gathered)}</p>")
        parts.append(f"</{block}>\n")
    parts.append("      </mesh>\n    </geometry>\n")
    return "".join(parts), used


def _set_of(key: str, base: str) -> str:
    """Return the set suffix of ``key`` under ``base``: ``"0"`` for ``base`` itself."""
    if key == base:
        return "0"
    if key.startswith(base + "_"):
        return key[len(base) + 1 :] or "0"
    return ""


def _gather(starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Return the indices of every ``conn[start:start + size]`` run, concatenated."""
    total = int(sizes.sum())
    run = np.repeat(np.arange(len(sizes)), sizes)
    heads = np.cumsum(sizes) - sizes
    return starts[run] + np.arange(total) - heads[run]


def _reachable(scene: SceneData, roots_per_scene: list[tuple[int, ...]]) -> set[int]:
    seen: set[int] = set()
    stack = [r for roots in roots_per_scene for r in roots]
    while stack:
        idx = stack.pop()
        if idx in seen or not (0 <= idx < len(scene.nodes)):
            continue
        seen.add(idx)
        stack.extend(scene.nodes[idx].children)
    return seen


def _controllers_xml(
    scene: SceneData, ids: _Ids, written: set[int], name: str
) -> dict[tuple[int, int], tuple[str, int | None, int]]:
    """Return ``(xml, skeleton root, slot)`` per (skin, mesh) a written node instances."""
    skins = scene.global_attrs.get("skins", [])
    out: dict[tuple[int, int], tuple[str, int | None, int]] = {}
    for n, node in enumerate(scene.nodes):
        s = node.extras.get("skin")
        if (
            n not in written
            or not isinstance(s, int)
            or node.mesh is None
            or (s, node.mesh) in out
        ):
            continue
        if not (0 <= s < len(skins)):
            _warn(
                f"{name!r}: a node names skin {s}, which the scene lacks; it is written unskinned."
            )
            continue
        skin = skins[s]
        mesh = scene.meshes[node.mesh]
        joints = mesh.vertex_attrs.get("joints")
        weights = mesh.vertex_attrs.get("weights")
        if joints is None or weights is None:
            _warn(
                f"{name!r}: skin {s} covers mesh {node.mesh}, which has no joints/weights; it is written unskinned."
            )
            continue
        joint_nodes = list(skin.get("joints", []))
        joints = np.asarray(joints)
        weights = np.asarray(weights, dtype=np.float64)
        if (
            joints.shape != weights.shape
            or joints.ndim != 2
            or len(joints) != len(mesh.vertices)
        ):
            raise CodecError(
                f"{name!r}: mesh {node.mesh} joints and weights are not both (n_vertices, k)."
            )
        live = weights > 0
        if live.any() and (
            joints[live].min() < 0 or joints[live].max() >= len(joint_nodes)
        ):
            raise CodecError(
                f"{name!r}: skin {s} has a joint index past its {len(joint_nodes)} joints."
            )
        sid = ids.skin(len(out))
        ibm = np.asarray(
            skin.get(
                "inverse_bind_matrices", np.tile(np.eye(4), (len(joint_nodes), 1, 1))
            ),
            dtype=np.float64,
        )
        if ibm.size != 16 * len(joint_nodes):
            raise CodecError(
                f"{name!r}: skin {s} holds {ibm.size // 16} inverse bind matrices "
                f"for {len(joint_nodes)} joints."
            )
        ibm = ibm.reshape(len(joint_nodes), 16)
        bsm = np.asarray(skin.get("bind_shape_matrix", np.eye(4)), dtype=np.float64)
        if bsm.size != 16:
            raise CodecError(
                f"{name!r}: skin {s} has a bind_shape_matrix of {bsm.size} values, not 16."
            )
        unwritten = [
            j for j in joint_nodes if 0 <= j < len(scene.nodes) and j not in written
        ]
        if unwritten:
            _warn(
                f"{name!r}: skin {s} names {len(unwritten)} joint node(s) that no "
                "scene reaches; they are written as missing."
            )
        names = _joint_names(ids, joint_nodes, written)
        vcount = live.sum(axis=1)
        flat_w = weights[live]
        flat_j = joints[live]
        v = (
            np.stack([flat_j, np.arange(len(flat_w))], axis=1).ravel()
            if len(flat_w)
            else np.zeros(0, np.int64)
        )
        out[(s, node.mesh)] = (
            f'    <controller id="{sid}"{_name_attr(str(skin.get("name", "")))}>\n'
            f'      <skin source="#{ids.geometry(node.mesh)}">\n'
            f"        <bind_shape_matrix>{_nums(bsm)}</bind_shape_matrix>\n"
            + _name_source_xml(f"{sid}-joints", names, "JOINT")
            + _source_xml(f"{sid}-bind_poses", ibm, ("TRANSFORM",))
            + _source_xml(f"{sid}-weights", flat_w.reshape(-1, 1), ("WEIGHT",))
            + f'        <joints><input semantic="JOINT" source="#{sid}-joints"/><input semantic="INV_BIND_MATRIX" source="#{sid}-bind_poses"/></joints>\n'
            f'        <vertex_weights count="{len(vcount)}"><input semantic="JOINT" source="#{sid}-joints" offset="0"/><input semantic="WEIGHT" source="#{sid}-weights" offset="1"/>'
            f"<vcount>{_nums(vcount)}</vcount><v>{_nums(v)}</v></vertex_weights>\n"
            "      </skin>\n    </controller>\n",
            _skeleton(scene, joint_nodes, written),
            len(out),
        )
    return out


def _joint_names(ids: _Ids, joint_nodes: list[int], written: set[int]) -> list[str]:
    """Return one ``Name_array`` entry per joint, each resolving to its own node.

    A joint is named by its node's sid, which is what a reader looks up
    first; when two joints of one skin share a sid, the second and later
    ones fall back to their node id, which is unique in the document.
    """
    sids = [ids.sid(j) if j in written else None for j in joint_nodes]
    seen: set[str] = set()
    names: list[str] = []
    for k, (j, sid) in enumerate(zip(joint_nodes, sids, strict=True)):
        if sid is None:
            names.append(f"missing{k}")
        elif sid in seen:
            names.append(ids.node(j))
        else:
            seen.add(sid)
            names.append(sid)
    return names


def _spelled(node: SceneNode) -> tuple[list[dict[str, Any]], bool]:
    """Return the transform elements to spell for a node, and whether they are its own.

    The elements of ``extras["transforms"]`` are kept when they still compose
    to ``node.matrix``; otherwise one ``<matrix sid="transform">`` stands in
    and the second value is False.
    """
    spelled = node.extras.get("transforms")
    if isinstance(spelled, list) and spelled:
        try:
            matrix = np.eye(4)
            for t in spelled:
                values = np.asarray(t["values"], dtype=np.float64)
                if t["kind"] not in _KIND_SIZE or values.size != _KIND_SIZE[t["kind"]]:
                    raise ValueError
                matrix = matrix @ _matrix_of(t["kind"], values.ravel())
        except (KeyError, TypeError, ValueError):
            matrix = None
        if matrix is not None and np.allclose(matrix, node.matrix, atol=1e-9):
            return [{**t, "sid": _transform_sid(t.get("sid"))} for t in spelled], True
    return [
        {
            "kind": "matrix",
            "sid": "transform",
            "values": np.asarray(node.matrix, dtype=np.float64).ravel().tolist(),
        }
    ], False


def _roots(scene: SceneData) -> tuple[int, ...]:
    children: set[int] = set()
    for node in scene.nodes:
        children.update(node.children)
    return tuple(i for i in range(len(scene.nodes)) if i not in children)


def _node_xml(
    scene: SceneData,
    ids: _Ids,
    idx: int,
    symbols_of: list[list[int]],
    controllers: dict[tuple[int, int], tuple[str, int | None, int]],
    stack: tuple[int, ...],
    depth: int,
    name: str,
) -> str:
    if idx in stack:
        raise CodecError(f"{name!r}: the node graph has a cycle through node {idx}.")
    if not (0 <= idx < len(scene.nodes)):
        raise CodecError(f"{name!r}: node index {idx} is out of range.")
    node = scene.nodes[idx]
    pad = "  " * depth
    joint = ' type="JOINT"' if node.extras.get("type") == "JOINT" else ""
    parts = [
        f'{pad}<node id="{ids.node(idx)}" sid="{ids.sid(idx)}"{_name_attr(node.name)}{joint}>\n'
    ]
    spelled, own = _spelled(node)
    if own and any(
        t["sid"] is None and node.extras["transforms"][k].get("sid")
        for k, t in enumerate(spelled)
    ):
        _warn(
            f"{name!r}: node {idx} spells a transform sid that is not an XML name; "
            "it is written without one."
        )
    for t in spelled:
        sid = f' sid="{t["sid"]}"' if t.get("sid") else ""
        parts.append(
            f"{pad}  <{t['kind']}{sid}>{_nums(np.asarray(t['values']))}</{t['kind']}>\n"
        )
    if node.mesh is not None:
        if not (0 <= node.mesh < len(scene.meshes)):
            raise CodecError(
                f"{name!r}: node {idx} names mesh {node.mesh}, which the scene lacks."
            )
        bind = ""
        if symbols_of[node.mesh]:
            bind = (
                "<bind_material><technique_common>"
                + "".join(
                    f'<instance_material symbol="{ids.material(m)}" target="#{ids.material(m)}"/>'
                    for m in symbols_of[node.mesh]
                )
                + "</technique_common></bind_material>"
            )
        s = node.extras.get("skin")
        key = (s, node.mesh) if isinstance(s, int) else None
        if key in controllers:
            _, skeleton, slot = controllers[key]
            skel = (
                f"<skeleton>#{ids.node(skeleton)}</skeleton>"
                if skeleton is not None
                else ""
            )
            parts.append(
                f'{pad}  <instance_controller url="#{ids.skin(slot)}">{skel}{bind}</instance_controller>\n'
            )
        else:
            parts.append(
                f'{pad}  <instance_geometry url="#{ids.geometry(node.mesh)}">{bind}</instance_geometry>\n'
            )
    parts.extend(
        _node_xml(
            scene, ids, child, symbols_of, controllers, (*stack, idx), depth + 1, name
        )
        for child in node.children
    )
    parts.append(f"{pad}</node>\n")
    return "".join(parts)


def _skeleton(
    scene: SceneData, joint_nodes: list[int], written: set[int]
) -> int | None:
    """Return the written root above the first written joint."""
    joints = [j for j in joint_nodes if j in written]
    if not joints:
        return None
    parent: dict[int, int] = {}
    for i, node in enumerate(scene.nodes):
        for c in node.children:
            parent.setdefault(c, i)
    top = joints[0]
    seen = {top}
    while top in parent and parent[top] not in seen and parent[top] in written:
        top = parent[top]
        seen.add(top)
    return top


_PARAMS_OF_WIDTH = {
    1: ("X",),
    2: ("X", "Y"),
    3: ("X", "Y", "Z"),
    4: ("X", "Y", "Z", "W"),
    16: ("TRANSFORM",),
}


def _animations_xml(scene: SceneData, ids: _Ids, written: set[int], name: str) -> str:
    out: list[str] = []
    for a, anim in enumerate(scene.global_attrs.get("animations", [])):
        aid = ids.animation(a)
        samplers = anim.get("samplers", [])
        body: list[str] = []
        emitted: set[int] = set()
        for c, channel in enumerate(anim.get("channels", [])):
            target = channel.get("target", {})
            node_idx = target.get("node")
            s = channel.get("sampler")
            if (
                not isinstance(node_idx, int)
                or node_idx not in written
                or not isinstance(s, int)
                or not (0 <= s < len(samplers))
            ):
                _warn(
                    f"{name!r}: animation {a} channel {c} names a node or sampler "
                    "the written scene lacks; it is dropped."
                )
                continue
            spelled, _ = _spelled(scene.nodes[node_idx])
            sid = target.get("sid")
            path = target.get("path")
            hit = None
            for t in spelled:
                if (sid is not None and t.get("sid") == sid) or (
                    sid is None
                    and path is not None
                    and t["kind"] == _KIND_OF_PATH.get(path)
                ):
                    hit = t
                    break
            if hit is None or not hit.get("sid"):
                _warn(
                    f"{name!r}: animation {a} channel {c} targets {sid or path!r} on node "
                    f"{node_idx}, which is written without such a transform; it is dropped."
                )
                continue
            base = f"{aid}-s{s}"
            member = target.get("member")
            suffix = (
                ""
                if not member
                else (member if member.startswith("(") else f".{member}")
            )
            if s not in emitted:
                emitted.add(s)
                body.append(_sampler_xml(samplers[s], base, a, s, name))
            body.append(
                f'      <channel source="#{base}-sampler" target="{ids.node(node_idx)}/{hit["sid"]}{suffix}"/>\n'
            )
        if body:
            out.append(
                f'    <animation id="{aid}"{_name_attr(str(anim.get("name", "")))}>\n'
                + "".join(body)
                + "    </animation>\n"
            )
    return "".join(out)


def _sampler_xml(sampler: dict[str, Any], base: str, a: int, s: int, name: str) -> str:
    """Return the sources and ``<sampler>`` of one animation sampler."""
    times = np.asarray(sampler["times"], dtype=np.float64).ravel()
    values = np.asarray(sampler["values"], dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if len(times) != len(values):
        raise CodecError(
            f"{name!r}: animation {a} sampler {s} has {len(times)} times but {len(values)} values."
        )
    width = values.shape[1]
    params = _PARAMS_OF_WIDTH.get(width, tuple(f"C{k}" for k in range(width)))
    interp = str(sampler.get("interpolation", "LINEAR"))
    if interp not in _INTERPOLATIONS:
        raise CodecError(
            f"{name!r}: animation {a} sampler {s} has interpolation {interp!r}; "
            f"COLLADA names {', '.join(_INTERPOLATIONS)}."
        )
    body: list[str] = []
    inputs = [
        f'<input semantic="INPUT" source="#{base}-input"/>',
        f'<input semantic="OUTPUT" source="#{base}-output"/>',
        f'<input semantic="INTERPOLATION" source="#{base}-interpolation"/>',
    ]
    body.append(_source_xml(f"{base}-input", times.reshape(-1, 1), ("TIME",), indent=6))
    body.append(_source_xml(f"{base}-output", values, params, indent=6))
    body.append(
        _name_source_xml(
            f"{base}-interpolation", [interp] * len(times), "INTERPOLATION", indent=6
        )
    )
    for key, semantic in (
        ("in_tangents", "IN_TANGENT"),
        ("out_tangents", "OUT_TANGENT"),
    ):
        tangents = sampler.get(key)
        if tangents is not None:
            tangents = np.asarray(tangents, dtype=np.float64)
            if tangents.ndim == 1:
                tangents = tangents.reshape(-1, 1)
            tp = _PARAMS_OF_WIDTH.get(
                tangents.shape[1], tuple(f"C{k}" for k in range(tangents.shape[1]))
            )
            body.append(_source_xml(f"{base}-{key}", tangents, tp, indent=6))
            inputs.append(f'<input semantic="{semantic}" source="#{base}-{key}"/>')
    body.append(f'      <sampler id="{base}-sampler">{"".join(inputs)}</sampler>\n')
    return "".join(body)
