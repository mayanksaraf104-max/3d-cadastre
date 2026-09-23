import os
import psycopg2
import trimesh
import numpy as np
import mapbox_earcut
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import explain_validity
import config

# ==============================================================
# WKT -> MESH
#
# The previous implementation parsed faces with
#     re.findall(r'\(\((.*?)\)\)', wkt_str)
# and then took `pts[:3]` from each match. Three separate defects came
# out of that, all of which silently produced a WRONG model rather than
# an error:
#
#   1. `pts[:3]` keeps only the first three vertices of every face. Any
#      face that is not already a triangle -- every rectangular wall, every
#      floor slab, every roof panel of an extruded storey -- was truncated
#      to a triangle spanning three of its corners. The exported building
#      was therefore not the building registered in the ledger.
#
#   2. The regex cannot see interior rings. A face with a hole (a courtyard
#      slab, a light well, a lift shaft) parses as though the hole were a
#      separate solid face, so voids rendered as filled material -- exactly
#      the holes that ai_to_ogc.py and main.py take care to preserve all the
#      way through extrusion.
#
#   3. It cannot distinguish a face boundary from a ring boundary at all,
#      so POLYHEDRALSURFACE and TIN were handled by luck rather than by
#      structure.
#
# This parser walks the WKT by paren depth instead. Which depth is a face and
# which is a ring depends on the geometry type (see _NESTING below); within a
# face the first ring is the exterior and the rest are holes.
# ==============================================================

# The database contract for authoritative solids is exactly
# `POLYHEDRALSURFACE Z`:  (((ring),(hole)),((ring)))
#   -> face at depth 2, rings at depth 3, every vertex exactly x y z.
# Every other type/dimension (TIN, MULTIPOLYGON, POLYGON, M, ZM, bare 2D...) is
# rejected rather than guessed at: their semantics are not the ledger's, and a
# fourth coordinate would otherwise be silently discarded.
_NESTING = {
    "POLYHEDRALSURFACE": (2, 3),
}
SUPPORTED_TYPES = tuple(_NESTING)

# Maximum distance (in the ledger's SRID units, i.e. metres for a projected
# CRS) any vertex of a face may sit from the face's plane. Faces beyond this
# are rejected, never flattened.
PLANARITY_TOLERANCE = 1e-4

# Triangulated area must match the projected polygon's area to this relative
# tolerance, otherwise the triangulation is treated as failed.
_AREA_REL_TOLERANCE = 1e-6


def parse_wkt_faces(wkt_str):
    """
    Returns [[ring, ring, ...], ...] -- a list of faces, each a list of rings,
    each ring a list of (x, y, z) tuples. The first ring of a face is its
    exterior; any further rings are holes.

    Fails closed: anything other than `POLYHEDRALSURFACE Z`, unexpected
    nesting, unbalanced parentheses, or any ring that cannot be parsed in full
    (including a vertex that is not exactly x y z) returns [] for the WHOLE
    geometry.
    Nothing is dropped or patched up, so an exterior ring can never be lost
    and leave a hole standing in for it.
    """
    text = wkt_str.strip()
    body_start = text.find("(")
    if body_start == -1:
        return []

    header = text[:body_start].upper().split()
    if not header:
        return []
    geom_type, dims = header[0], header[1:]
    if geom_type not in _NESTING or dims != ["Z"]:
        return []
    face_depth, ring_depth = _NESTING[geom_type]

    faces, current_face, buf = [], None, []
    depth = 0

    for ch in text[body_start:]:
        if ch == "(":
            depth += 1
            if depth > ring_depth:
                return []
            if depth == face_depth:
                current_face = []
            elif depth == ring_depth:
                buf = []
            continue

        if ch == ")":
            if depth == ring_depth:
                ring = _parse_ring("".join(buf))
                if ring is None:
                    return []
                current_face.append(ring)
            elif depth == face_depth:
                if not current_face:
                    return []
                faces.append(current_face)
                current_face = None
            depth -= 1
            if depth < 0:
                return []
            continue

        if depth == ring_depth:
            buf.append(ch)

    if depth != 0:
        return []

    return faces


def _parse_ring(ring_text):
    """Parses one ring; returns None unless EVERY vertex is exactly x y z
    (a missing Z or an extra coordinate such as M is never dropped)."""
    pts = []
    for token in ring_text.split(","):
        parts = token.strip().split()
        if len(parts) != 3:
            return None
        try:
            pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            return None
    # Drop the repeated closing vertex; triangulation wants an open ring.
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts if len(pts) >= 3 else None


def _face_normal(ring):
    """Newell's method -- robust for non-planar and non-convex rings, unlike
    the cross product of the first three edges (which degenerates the moment
    those three points happen to be collinear).

    Returns None for a degenerate (zero-area) ring: no normal is invented.
    Coordinates are centred first so the products stay small at projected-CRS
    magnitudes."""
    pts = np.asarray(ring, dtype=float)
    pts = pts - pts.mean(axis=0)
    n = np.zeros(3)
    for i, cur in enumerate(pts):
        nxt = pts[(i + 1) % len(pts)]
        n[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2])
        n[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0])
        n[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1])
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-12 else None


def _reject_face(reason):
    print(f"   ⚠️ face rejected: {reason}")
    return []


def triangulate_face(face_rings):
    """
    Triangulates one face (exterior ring + optional holes) into 3D triangles
    whose corners are the ledger's own vertices, unmodified.

    The face must be planar (within PLANARITY_TOLERANCE). It is projected onto
    its own plane, checked for validity there, and triangulated in 2D where
    holes and concavity are handled properly. Triangles index straight back
    into the original 3D vertices, so no coordinate is recomputed or rounded.

    Fails closed: a degenerate, non-planar or invalid face, or a triangulation
    that does not reproduce the projected polygon's area, yields [] -- the face
    is never repaired (no buffer(0)), flattened, or fan-triangulated.
    """
    exterior = face_rings[0]

    normal = _face_normal(exterior)
    if normal is None:
        return _reject_face("degenerate (zero-area) exterior ring")

    all_pts = [p for ring in face_rings for p in ring]
    pts_arr = np.array(all_pts, dtype=float)
    origin = np.array(exterior, dtype=float).mean(axis=0)

    deviation = float(np.abs((pts_arr - origin) @ normal).max())
    if deviation > PLANARITY_TOLERANCE:
        return _reject_face(
            f"non-planar (max deviation {deviation:.6f} from its plane, "
            f"tolerance {PLANARITY_TOLERANCE})"
        )

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    def to_2d(ring):
        arr = np.array(ring, dtype=float) - origin
        return np.column_stack((arr @ u, arr @ v))

    rings_2d = [to_2d(r) for r in face_rings]

    try:
        poly_2d = ShapelyPolygon(rings_2d[0], rings_2d[1:])
        if not poly_2d.is_valid:
            return _reject_face(f"invalid projected polygon ({explain_validity(poly_2d)})")
        if poly_2d.is_empty or poly_2d.area <= 0.0:
            return _reject_face("empty projected polygon")

        verts_2d = np.vstack(rings_2d)
        ring_ends = np.cumsum([len(r) for r in rings_2d]).astype(np.uint32)
        tris = np.asarray(
            mapbox_earcut.triangulate_float64(verts_2d, ring_ends), dtype=np.int64
        ).reshape(-1, 3)
        if len(tris) == 0:
            return _reject_face("triangulation produced no triangles")

        a, b, c = verts_2d[tris[:, 0]], verts_2d[tris[:, 1]], verts_2d[tris[:, 2]]
        tri_area = 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        ).sum()
        if abs(tri_area - poly_2d.area) > _AREA_REL_TOLERANCE * poly_2d.area:
            return _reject_face(
                f"triangulated area {tri_area:.6f} != face area {poly_2d.area:.6f}"
            )
    except Exception as exc:
        return _reject_face(f"triangulation failed ({exc})")

    return [[all_pts[i] for i in tri] for tri in tris]


# ==============================================================
# COLOURING
# ==============================================================

def tier_color(tier_type, floor_level):
    """
    FIX: colour used to be chosen with `if "METRO" in unit_id` / `elif
    "Floor_2" in unit_id` -- string matching on an identifier. That broke the
    moment a unit was named anything else, and it recognised floor 2
    specifically, so a ten-storey building rendered as one undifferentiated
    green mass. tier_type and floor_level are real columns on
    property_registry; use them.
    """
    if tier_type == "SUBSURFACE":
        return [90, 120, 255, 170]           # translucent blue
    if tier_type == "AIR_RIGHTS" or (floor_level or 0) > 0:
        level = max(1, int(floor_level or 1))
        t = min(1.0, level / 12.0)           # warm gradient climbing the building
        return [255, int(200 - 90 * t), int(60 + 40 * t), 205]
    return [100, 220, 120, 255]              # opaque green surface parcel


# ==============================================================
# EXPORT
# ==============================================================

def export_postgis_to_glb(output_filename="approved_cadastre.glb"):
    print("🌍 Connecting to PostGIS to query registered legal entities...")
    conn = cursor = None
    try:
        conn = psycopg2.connect(**config.PG_DSN_KWARGS)
        cursor = conn.cursor()

        # Pull the derived volumetric columns too (see db_engine.py) so the
        # exported model carries, per node, what the ledger actually measured
        # -- rather than a shape with no provenance attached.
        cursor.execute("""
            SELECT unit_id, ST_AsText(boundary), tier_type, floor_level,
                   volume_m3, z_min, z_max, ulpin
            FROM property_registry
            ORDER BY tier_type, floor_level, unit_id;
        """)
        rows = cursor.fetchall()

        if not rows:
            print("⚠️ No registered properties found in the database ledger.")
            return

        # ----------------------------------------------------------
        # ONE offset for the ENTIRE scene.
        #
        # FIX: `global_offset` was declared inside the per-row loop and reset
        # to None for every property, so each mesh was recentred on its OWN
        # first vertex. Every building in the national ledger was therefore
        # translated to the origin independently -- the exported .glb showed
        # them all stacked on top of one another, with every spatial
        # relationship between titles destroyed. That is catastrophic for a
        # cadastre viewer, and it looks plausible on screen, which is why it
        # could survive review: you see buildings, just not where they are.
        #
        # The offset exists only to keep large projected coordinates inside
        # float32 range for WebGL, so it must be a SINGLE scene-wide
        # translation, recorded below so a viewer can invert it.
        # ----------------------------------------------------------
        all_faces = {}
        sum_x = sum_y = 0.0
        vertex_count = 0

        for row in rows:
            unit_id, wkt_str = row[0], row[1]
            faces = parse_wkt_faces(wkt_str or "")
            if not faces:
                # Never omit a legal unit: abort the whole export instead.
                raise ValueError(
                    f"authoritative geometry for {unit_id} is not a parsable "
                    f"POLYHEDRALSURFACE Z -- export aborted, no file written"
                )
            all_faces[unit_id] = (faces, row)
            for face in faces:
                for pt in face[0]:
                    sum_x += pt[0]
                    sum_y += pt[1]
                    vertex_count += 1

        if not vertex_count:
            print("⚠️ No parsable geometry in the ledger.")
            return

        # Centroid rather than "first vertex seen": centring the scene keeps
        # the largest coordinate magnitude as small as possible, which is the
        # actual goal of the shift.
        global_offset = np.array([sum_x / vertex_count, sum_y / vertex_count, 0.0])
        print(f"📐 Scene-wide WebGL offset (applied once): "
              f"X {global_offset[0]:.2f}, Y {global_offset[1]:.2f}")

        scene = trimesh.Scene()
        exported = skipped = 0

        for unit_id, (faces, row) in all_faces.items():
            tier_type, floor_level = row[2], row[3]
            volume_m3, z_min, z_max, ulpin = row[4], row[5], row[6], row[7]

            vertices, tri_faces, vertex_map = [], [], {}
            rejected_faces = 0
            degenerate_tris = 0

            for face in faces:
                face_tris = triangulate_face(face)
                if not face_tris:
                    rejected_faces += 1
                    continue
                for tri in face_tris:
                    idx = []
                    for pt in tri:
                        # No rounding: float64 is kept exactly as the ledger
                        # stored it (the GLB writer narrows to float32 itself).
                        shifted = (float(pt[0]) - float(global_offset[0]),
                                   float(pt[1]) - float(global_offset[1]),
                                   float(pt[2]))
                        if shifted not in vertex_map:
                            vertex_map[shifted] = len(vertices)
                            vertices.append(shifted)
                        idx.append(vertex_map[shifted])
                    # A degenerate triangle means the face is incomplete; it
                    # is never dropped silently (the unit is rejected below).
                    if len(set(idx)) == 3:
                        tri_faces.append(idx)
                    else:
                        degenerate_tris += 1

            if rejected_faces:
                # An incomplete solid is not the title in the ledger.
                print(f"   ⚠️ {unit_id}: {rejected_faces} of {len(faces)} face(s) "
                      f"rejected — unit not exported.")
                skipped += 1
                continue

            if degenerate_tris:
                print(f"   ⚠️ {unit_id}: {degenerate_tris} degenerate triangle(s) "
                      f"— unit not exported.")
                skipped += 1
                continue

            if not tri_faces:
                print(f"   ⚠️ {unit_id} produced no renderable triangles — skipped.")
                skipped += 1
                continue

            mesh = trimesh.Trimesh(
                vertices=np.array(vertices, dtype=np.float64),
                faces=np.array(tri_faces, dtype=np.int64),
                process=False,     # keep the ledger's exact vertices
            )
            mesh.visual.face_colors = tier_color(tier_type, floor_level)

            # Provenance travels with the geometry.
            mesh.metadata.update({
                "ulpin": ulpin,
                "unit_id": unit_id,
                "tier_type": tier_type,
                "floor_level": floor_level,
                "volume_m3": float(volume_m3) if volume_m3 is not None else None,
                "z_min": float(z_min) if z_min is not None else None,
                "z_max": float(z_max) if z_max is not None else None,
            })

            scene.add_geometry(mesh, node_name=unit_id)
            exported += 1
            print(f"📦 {unit_id} [{tier_type}, floor {floor_level}]: "
                  f"{len(tri_faces)} triangles")

        # Record the shift so a viewer can map a picked vertex back to a real
        # coordinate. Without this the exported model is unreferenced.
        scene.metadata.update({
            "cadastre_srid": config.CADASTRE_SRID,
            "global_offset_x": float(global_offset[0]),
            "global_offset_y": float(global_offset[1]),
            "note": "Add the offsets back to vertex XY to recover SRID coordinates.",
        })

        scene.export(output_filename)
        print(f"✨ Exported {exported} legal title(s) to {output_filename}"
              + (f" ({skipped} skipped)" if skipped else ""))

    except Exception as e:
        print(f"❌ Export Failed: {e}")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    export_postgis_to_glb()