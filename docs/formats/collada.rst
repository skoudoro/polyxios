.. _format-collada:

COLLADA
=======

.. rst-class:: px-badges

``.dae`` ``read + write`` ``eager`` ``scene: read_scene + write_scene``

Summary of the specification
-----------------------------

COLLADA is the Khronos Group's XML interchange format for 3D scenes, the file every modelling tool of the 2000s learned to export and the one glTF was designed to replace. A document is a set of libraries tied together by ``#id`` references: geometries, effects and the materials that instance them, images, controllers (skins and morphs), animations and visual scenes, with a ``<scene>`` naming the visual scene to show. A ``<geometry>`` holds a ``<mesh>`` of ``<source>`` arrays, each read through an ``<accessor>`` that names its stride and parameters, a ``<vertices>`` element binding the position source, and primitive blocks - ``<triangles>``, ``<polylist>``, ``<polygons>``, ``<lines>``, ``<linestrips>``, ``<tristrips>``, ``<trifans>`` - whose ``<p>`` index tuples name one entry of every ``<input>`` per corner, so a corner carries its own normal and texture coordinate. Nodes hold transforms as ``<matrix>``, ``<translate>``, ``<rotate>``, ``<scale>`` and ``<lookat>`` elements applied in document order, each addressable by ``sid`` so an ``<animation>`` channel can target it. Effects describe materials in fixed-function terms - phong, lambert, blinn or constant shading with emission, diffuse, specular and transparency channels that are colours or textures reached through a sampler and a surface.

Specification at a glance
--------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - document
     - UTF-8 XML, root ``<COLLADA version="1.4.1">`` (or 1.5.0) in the ``collada.org/2005/11/COLLADASchema`` namespace
   * - source
     - ``<float_array count>`` (or ``Name_array``) plus ``<accessor count stride offset>`` with one ``<param name type>`` per used component
   * - mesh
     - ``<vertices>`` binding ``POSITION``; primitive blocks with ``<input semantic offset set>`` and ``<p>`` index tuples, ``<vcount>`` for a polylist
   * - primitive blocks
     - ``triangles``, ``polylist``, ``polygons`` (with ``<ph>`` holes), ``lines``, ``linestrips``, ``tristrips``, ``trifans``
   * - node transforms
     - ``<matrix>`` (16, row-major), ``<translate>`` (3), ``<rotate>`` (axis + degrees), ``<scale>`` (3), ``<lookat>`` (9), ``<skew>``; composed in document order
   * - instancing
     - ``<instance_geometry>`` with ``<bind_material>``, ``<instance_controller>`` with ``<skeleton>``, ``<instance_node>``, cameras and lights
   * - effect
     - ``<profile_COMMON>`` with ``<newparam>`` surfaces and samplers, a ``<technique>`` of ``phong`` / ``lambert`` / ``blinn`` / ``constant``
   * - skin
     - ``<bind_shape_matrix>``, ``<joints>`` (``JOINT`` names, ``INV_BIND_MATRIX``), ``<vertex_weights>`` with ``<vcount>`` and ``<v>`` joint/weight pairs
   * - animation
     - ``<sampler>`` of ``INPUT`` times, ``OUTPUT`` values, ``INTERPOLATION`` names and tangents; ``<channel target="node/sid.MEMBER">``
   * - asset
     - ``<up_axis>`` X_UP / Y_UP / Z_UP, ``<unit name meter>``, contributor, created and modified

.. rst-class:: px-speclink

`Read the full COLLADA 1.4.1 specification ↗ <https://www.khronos.org/files/collada_spec_1_4.pdf>`__

Reading
-------

``read_scene`` returns a :class:`~polyxios.SceneData` with one mesh per ``<geometry>``, in library order:

.. code-block:: python

    import polyxios as px

    scene = px.read_scene("robot.dae")
    scene.meshes       # tuple of PolyData, one per <geometry>
    scene.nodes        # tuple of SceneNode: local matrix, mesh index, children
    scene.scenes       # root node indices of each <visual_scene>
    scene.materials    # tuple of SceneMaterial built from the effects
    scene.textures     # sampler settings, one per distinct (image, wrap, filter)
    scene.images       # <library_images>: a uri, or bytes from <hex> / data: URIs

A corner is one index per ``<input>``; corners that name the same position but a different normal, texture coordinate or colour become separate vertices, in first-occurrence order, the way an OBJ reader splits ``v/vt/vn`` triples. A block whose inputs all share one offset is read as values per position and keeps every position, named or not; a later block naming another source for the same attribute splits its corners like any other disagreeing input. When one block carries an input another block of the same mesh lacks (lines beside textured triangles, say), the corners without it get zeros, with a warning. ``NORMAL`` lands in ``normals``, ``TEXCOORD`` set 0 in ``texcoords`` and set *n* in ``texcoords_n``, ``COLOR`` in ``colors`` (3 or 4 columns as the file has them); any other semantic keeps its lowercased name. Every primitive block maps to the VTK element it is: triangles, quads and polygons (a polylist entry by its size), lines, ``poly_line`` for a line strip, ``triangle_strip`` for a triangle strip, and a fan is expanded into triangles. A block's ``material`` symbol is resolved through the instancing node's ``<bind_material>``, or straight to a material of that id, into ``element_attrs["material"]`` (``-1`` where none). A ``<geometry name>`` is the mesh's ``global_attrs["mesh_name"]``, the key glTF and MED use, and every ``<library_*>`` of a kind is read, not only the first.

Nodes keep their ``id``, ``sid`` and ``JOINT`` type in ``extras``, along with the transform elements they were spelled with (``extras["transforms"]``, a list of ``{"kind", "sid", "values"}``), which is what lets an animation find its target again on write. A node instancing several geometries gets a child node per extra one; an ``<instance_node>`` is copied into place. A skin (``<instance_controller>``) attaches ``joints`` (int32, ``(n, 4)``) and ``weights`` (``(n, 4)``) to its mesh - the four heaviest influences when a vertex has more, renormalised with a warning, and zeros for the vertices a short ``<vertex_weights>`` leaves out, also with a warning - and puts the controller under ``global_attrs["skins"]`` with its ``joints`` resolved to node indices by ``sid`` - searched under the instance's ``<skeleton>`` roots first, then by ``id`` under those roots for the exporters that name joints so, and only then anywhere in the document, so two rigs sharing joint names each bind their own - its ``inverse_bind_matrices`` (``(k, 4, 4)``) and ``bind_shape_matrix``. Animations land under ``global_attrs["animations"]`` in the same shape glTF uses: each with ``channels`` (``sampler`` index and a ``target`` of node index, ``path``, the transform's ``sid`` and an optional ``member`` such as ``ANGLE``) and ``samplers`` (``times``, ``values``, the COLLADA ``interpolation`` name and Bezier ``in_tangents`` / ``out_tangents`` when present). The ``<asset>`` up axis, unit, authoring tool and timestamps are ``global_attrs["asset"]``; the axis is recorded, never applied.

Flatten to a single :class:`~polyxios.PolyData`, or let :func:`~polyxios.read` do it with a warning:

.. code-block:: python

    mesh = scene.to_polydata()    # applies every node's world transform
    mesh = px.read("robot.dae")   # the same, with a UserWarning

Writing
-------

``write_scene`` spells a :class:`~polyxios.SceneData` back in COLLADA 1.4.1 syntax:

.. code-block:: python

    px.write_scene(scene, "out.dae")
    px.write(mesh, "out.dae")     # a one-node scene; material values become material_<n>

Each mesh becomes a ``<geometry>``, named by its ``global_attrs["mesh_name"]``, with its positions, ``normals``, ``texcoords``\ (``_n``) and ``colors`` as sources sharing one index; a set suffix that is not a number, or one another key of the mesh already spells, is written under the lowest free set number with a warning and reads back under that number. Triangles go in a ``<triangles>`` block, or into the ``<polylist>`` with the quads and polygons when the mesh has any; lines, line strips and triangle strips get their own blocks, and points and volume cells are skipped with a warning - a point cloud keeps its positions as a ``<mesh>`` without primitives, which is valid COLLADA and reads back as a mesh of no elements. Elements are grouped by material, each group its own block, in first-occurrence order: element order survives within one block, so a surface whose ``material`` column alternates reads back regrouped by material (``[0, 1, 0, 1]`` becomes ``[0, 0, 1, 1]``), and a mesh mixing lines with faces reads back with the faces first. Materials become ``<effect>`` entries in the shading their ``extras["shading"]`` names (phong by default) with the base colour as diffuse, ``emissive`` as emission, alpha below one as the alpha of an ``A_ONE`` ``<transparent>`` colour, a base colour texture as the diffuse texture, a normal texture as an FCOLLADA ``<bump>``, and ``double_sided`` as the GOOGLEEARTH extra. Images are written by uri, or as a ``data:`` URI when they hold bytes. A node is spelled with the transform elements in ``extras["transforms"]`` when they still compose to its matrix, else as one ``<matrix sid="transform">``; a node with a ``skin`` in its extras is instanced through a ``<controller>`` built from its mesh's ``joints`` and ``weights``. Only nodes a ``<visual_scene>`` reaches are written: a node in no scene is dropped with a warning, a skin joint or animation channel naming it likewise. An animation channel targets the transform element whose ``sid`` (or kind, for a ``matrix`` / ``translation`` / ``scale`` / ``rotation`` path) the written node has, and is dropped with a warning otherwise; a sampler shared by several channels is written once, and one whose ``interpolation`` is not among the six the specification names (``LINEAR``, ``BEZIER``, ``HERMITE``, ``CARDINAL``, ``BSPLINE``, ``STEP``) is refused rather than written into a ``<Name_array>`` no reader could parse.

Format-specific options
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 20 56
   :class: px-spec-table

   * - Option
     - Default
     - Effect
   * - ``up_axis``
     - ``None``
     - ``"X_UP"``, ``"Y_UP"`` or ``"Z_UP"``; overrides ``global_attrs["asset"]["up_axis"]``, which itself defaults to ``Y_UP``.
   * - ``unit``
     - ``None``
     - ``{"name": ..., "meter": ...}``; overrides the asset's unit, which defaults to one metre.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Every count is checked before anything is allocated: a ``float_array`` whose ``count`` disagrees with its content, an accessor reading past its array, a ``<p>`` whose length disagrees with the block's ``count`` or ``<vcount>``, an index past its source, a ``#url`` the document does not hold, a transform with the wrong number of values and a node instancing itself are each refused by name.
- The writer uses fixed timestamps (``1970-01-01T00:00:00Z`` unless the asset carries its own), so two writes of one scene are byte-identical.
- A name holding a line break or tab is written as a character reference, since a parser folds the literal character to a space, so it reads back unchanged; a node ``id`` or ``sid`` that is not an XML name is replaced by a generated one, and two joints of one skin sharing a ``sid`` are named by their ``id`` so each binds its own node.
- A diffuse texture replaces the diffuse colour, as the format has room for one or the other: a textured material reads back with a white base colour and its alpha. Metallic and roughness have no COLLADA spelling and are not written; every material reads back with ``metallic=0``.
- Transparency follows the ``<transparent opaque="...">`` mode (``A_ONE``, ``A_ZERO``, ``RGB_ZERO``, ``RGB_ONE``; any other spelling is read as ``A_ONE`` with a warning) times ``<transparency>``; an alpha below one reads as ``alpha_mode="BLEND"``. The one exception is ``A_ONE`` with a white transparent colour and ``<transparency>`` 0, which SketchUp, Google Earth and Blender before 2.8 write for an *opaque* material: read literally it would be invisible, so it is read as opaque with a warning (once per document, as a SketchUp export has hundreds), as other readers do. COLLADA has no alpha mask: ``alpha_mode`` is derived from the alpha on read, and ``MASK`` with its ``alpha_cutoff`` is not written.
- Non-finite values are written in XML's own spelling (``NaN``, ``INF``, ``-INF``) and read back as such.
- A 1.5 sampler naming its image with ``<instance_image>`` resolves like a 1.4 sampler-and-surface chain.
- Polygons with holes (``<ph>``) keep their outer ring and drop the holes with a warning; ``<skew>`` transforms are ignored with a warning; a morph controller reads as its base geometry.
- A file whose numbers use a decimal comma is refused as "not numbers", except the asset's ``meter``, which is read with a warning since one exporter spells only that one in the machine's locale.
- ``.zae`` (a zipped ``.dae``) and external ``file.dae#id`` references are not read.
- ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`: the geometry is XML text.

.. seealso::

   :doc:`gltf` - the format's successor, sharing the SceneData model.
   :doc:`index` - the full format table.
