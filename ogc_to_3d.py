from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods
from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Plane
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface

try:
    from OCC.Core.BRepTools import breptools
    _outer_wire = breptools.OuterWire
except ImportError:  # newer pythonocc: static methods carry an _s suffix
    from OCC.Core.BRepTools import BRepTools
    _outer_wire = BRepTools.OuterWire_s


class UnsupportedOGCRepresentationError(ValueError):
    """
    The B-Rep has no lossless representation in the OGC type this module emits.
    The OCC B-Rep stays the authoritative geometry; nothing is tessellated, chorded or planarized.
    """


def _fmt(pnt):
    # repr() round-trips the double exactly; fixed-decimal formatting would alter measured XYZ
    return f"{pnt.X()!r} {pnt.Y()!r} {pnt.Z()!r}"


def _reject_curve(curve_type):
    raise UnsupportedOGCRepresentationError(
        f"Unsupported edge curve type (GeomAbs_CurveType={curve_type}): only straight lines are "
        "serializable. Curved OGC types (CIRCULARSTRING, CURVEPOLYGON, ...) are not part of the "
        "db_engine contract (POLYHEDRALSURFACEZ only), and approximating the measured curve is not allowed."
    )


def _wire_to_ring(wire, face):
    """
    Serializes one B-Rep wire of `face` as a closed linear ring "(x y z, ..., x y z)", walking
    edges in wire order and keeping each vertex's XYZ exactly as OCC reports it.
    Every edge must be a straight line: a polyhedral-surface ring cannot carry a curve, and
    replacing one with its end vertices would alter the measured geometry, so it is rejected.
    """
    explorer = BRepTools_WireExplorer(wire, face)
    coords = []

    while explorer.More():
        edge = topods.Edge(explorer.Current())
        curve_type = BRepAdaptor_Curve(edge).GetType()
        if curve_type != GeomAbs_Line:
            raise UnsupportedOGCRepresentationError(
                f"Curved edge (GeomAbs_CurveType={curve_type}) in a face boundary: POLYHEDRALSURFACE Z "
                "cannot represent it losslessly, and curved OGC output (CURVEPOLYGON/MULTISURFACE) "
                "is not part of the downstream contract. Refusing to serialize."
            )
        coords.append(_fmt(BRep_Tool.Pnt(explorer.CurrentVertex())))
        explorer.Next()

    if len(coords) < 3:
        raise ValueError("Degenerate wire: fewer than 3 vertices.")

    coords.append(coords[0])  # Close the loop
    return f"({', '.join(coords)})"


def _face_count(shape):
    count = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def occ_to_wkt(shape, geometry_type="Solid"):
    """
    Serializes OpenCASCADE geometry to OGC WKT (3D) without geometric alteration.
    The OCC B-Rep remains the authoritative geometry; the WKT is emitted only when it is an
    exact view of it, otherwise UnsupportedOGCRepresentationError is raised.

    Solid / MultiSolid -> POLYHEDRALSURFACE Z, only if the shape has exactly one shell, every
      face is planar and every edge of every wire is a straight line. Each B-Rep face is one
      patch: its outer wire is the exterior ring and its remaining wires are that same patch's
      interior rings.
      Not emitted (rejected): several shells (voids, multiple solids), non-planar faces, curved
      edges. No tessellation, chording or planarization, and no curved OGC types (CURVEPOLYGON /
      MULTISURFACE) are produced.
    Edge  -> LINESTRING Z (straight line only); curved edges are rejected.
    Point -> POINT Z.
    """
    if geometry_type == "Solid" or geometry_type == "MultiSolid":
        # One POLYHEDRALSURFACE Z is one surface. db_engine builds a single solid from it
        # (ST_MakeSolid), so voids / several solids flattened into one patch list would not be
        # an exact representation of the B-Rep.
        shells = []
        shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
        while shell_explorer.More():
            shells.append(topods.Shell(shell_explorer.Current()))
            shell_explorer.Next()
        if len(shells) != 1:
            raise UnsupportedOGCRepresentationError(
                f"Expected exactly one shell, found {len(shells)}: a single POLYHEDRALSURFACE Z cannot "
                "represent voids or several solids without altering the measured B-Rep. "
                "Refusing to serialize."
            )
        # Only the shell's own faces are the surface; loose faces elsewhere in `shape` would be
        # neither part of it nor safe to drop silently.
        if _face_count(shape) != _face_count(shells[0]):
            raise UnsupportedOGCRepresentationError(
                "Shape contains faces outside its single shell: not representable as one "
                "POLYHEDRALSURFACE Z without altering the measured B-Rep. Refusing to serialize."
            )

        face_explorer = TopExp_Explorer(shells[0], TopAbs_FACE)
        patches = []
        
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())

            if BRepAdaptor_Surface(face).GetType() != GeomAbs_Plane:
                raise UnsupportedOGCRepresentationError(
                    "Non-planar face: POLYHEDRALSURFACE Z cannot represent it losslessly, and "
                    "tessellating or planarizing it would alter the measured B-Rep. "
                    "Refusing to serialize."
                )

            outer_wire = _outer_wire(face)
            if outer_wire.IsNull():
                raise ValueError("Face has no outer wire.")

            # Exterior ring first, then the face's own inner wires as holes of the same patch
            rings = [_wire_to_ring(outer_wire, face)]

            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            while wire_explorer.More():
                wire = topods.Wire(wire_explorer.Current())
                if not wire.IsSame(outer_wire):
                    rings.append(_wire_to_ring(wire, face))
                wire_explorer.Next()

            patches.append(f"({', '.join(rings)})")
            face_explorer.Next()

        if not patches:
            raise ValueError("Shape contains no faces.")
            
        return f"POLYHEDRALSURFACE Z ({', '.join(patches)})"

    elif geometry_type == "Edge":
        edge = topods.Edge(shape)
        curve = BRepAdaptor_Curve(edge)
        curve_type = curve.GetType()

        # Vertices come back with orientation composed with the edge: FORWARD = traversal start, REVERSED = end
        start = end = None
        vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex = vertex_explorer.Current()
            if vertex.Orientation() == TopAbs_FORWARD:
                start = BRep_Tool.Pnt(topods.Vertex(vertex))
            elif vertex.Orientation() == TopAbs_REVERSED:
                end = BRep_Tool.Pnt(topods.Vertex(vertex))
            vertex_explorer.Next()
        if start is None or end is None:
            raise ValueError("Edge has no resolvable start/end vertices.")
        
        if curve_type == GeomAbs_Line:
            return f"LINESTRING Z ({_fmt(start)}, {_fmt(end)})"
        else:
            _reject_curve(curve_type)

    elif geometry_type == "Point":
        vertex = topods.Vertex(shape)
        pnt = BRep_Tool.Pnt(vertex)
        return f"POINT Z ({_fmt(pnt)})"
        
    else:
        raise ValueError(f"Geometry type {geometry_type} routing not implemented.")