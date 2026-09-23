"""
Authoritative jurisdiction lookup from the Survey of India (SOI) ABDB
DISTRICT_BOUNDARY shapefile.

One district-layer lookup returns BOTH the state and the district: the
WGS84 (lat, lon) point is transformed into the shapefile's OWN CRS (read from
its .prj -- never assumed, never EPSG:7755) and matched against the district
polygons with an exact Shapely STRtree point-in-polygon query. There is no
state-first lookup, no nearest-polygon fallback and no guessing.

Input is WGS84 latitude/longitude only; Z is not an input and is ignored.
SOI codes and names are returned exactly as stored in the dataset (never
truncated, padded or invented). If no polygon contains the point (or the
match is ambiguous), `JurisdictionUnavailable` is raised.

Configuration (nothing machine-specific is hardcoded):
  * dataset path: `shapefile_path=` argument, else the environment variable
    SOI_DISTRICT_BOUNDARY_SHP.
  * attribute names: the four DBF field names holding the state LGD code,
    district LGD code, state name and district name. Override per field with
    the environment variables SOI_FIELD_STATE_LGD, SOI_FIELD_DISTRICT_LGD,
    SOI_FIELD_STATE_NAME, SOI_FIELD_DISTRICT_NAME (or `fields=`). The
    defaults in DEFAULT_FIELDS are NOT verified against your copy of the
    dataset: if a field is missing the loader stops and lists the fields the
    file actually has, so you can set the right names.

Requires: pyshp (`import shapefile`), shapely >= 2, pyproj.
"""
import math
import os
import threading
import warnings
from dataclasses import dataclass

DEFAULT_FIELDS = {
    "state_lgd": "STATE_LGD",
    "district_lgd": "DIST_LGD",
    "state_name": "STATE_UT",
    "district_name": "DISTRICT",
}
_FIELD_ENV = {
    "state_lgd": "SOI_FIELD_STATE_LGD",
    "district_lgd": "SOI_FIELD_DISTRICT_LGD",
    "state_name": "SOI_FIELD_STATE_NAME",
    "district_name": "SOI_FIELD_DISTRICT_NAME",
}
SHAPEFILE_ENV = "SOI_DISTRICT_BOUNDARY_SHP"


class JurisdictionUnavailable(LookupError):
    """No authoritative jurisdiction could be established for the point."""


class JurisdictionDataIncomplete(JurisdictionUnavailable):
    """A polygon contains the point, but a required LGD/name attribute is missing."""


@dataclass(frozen=True)
class _IncompleteDistrict:
    attrs: tuple  # ((field, value as stored), ...)


@dataclass(frozen=True)
class DistrictJurisdiction:
    state_lgd: str
    district_lgd: str
    state_name: str
    district_name: str


def _code(value, field):
    """A code exactly as stored: strings untouched, integral numbers as their
    digits (no padding/truncation). Anything else is an error, not a guess."""
    if isinstance(value, str):
        if not value:
            raise ValueError(f"empty value in SOI field {field!r}")
        return value
    if isinstance(value, bool):
        raise ValueError(f"unexpected boolean in SOI field {field!r}")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    raise ValueError(f"unusable value {value!r} in SOI field {field!r}")


def _missing(value):
    """True for an absent/empty attribute value (nothing is filled in)."""
    return value is None or (isinstance(value, str) and not value.strip())


def _name(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"unusable name {value!r} in SOI field {field!r}")
    return value


class SOIDistrictResolver:
    """Loads the SOI district layer once and answers point lookups."""

    def __init__(self, shapefile_path, fields=None):
        import shapefile  # pyshp
        import shapely
        from pyproj import CRS, Transformer
        from shapely.geometry import shape
        from shapely.strtree import STRtree

        path = os.fspath(shapefile_path)
        base, _ = os.path.splitext(path)
        prj_path = base + ".prj"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SOI district shapefile not found: {path}")
        if not os.path.isfile(prj_path):
            raise FileNotFoundError(
                f"SOI shapefile has no .prj ({prj_path}); its CRS is never assumed.")
        with open(prj_path, "r", encoding="utf-8", errors="replace") as fh:
            self._crs = CRS.from_wkt(fh.read())
        self._to_soi = Transformer.from_crs("EPSG:4326", self._crs, always_xy=True)

        wanted = dict(DEFAULT_FIELDS)
        for key, env in _FIELD_ENV.items():
            if os.environ.get(env):
                wanted[key] = os.environ[env]
        wanted.update(fields or {})

        geoms, records = [], []
        skipped = 0
        with shapefile.Reader(path) as reader:
            available = {f[0].lower(): f[0] for f in reader.fields[1:]}
            missing = [n for n in wanted.values() if n.lower() not in available]
            if missing:
                raise KeyError(
                    f"SOI shapefile lacks field(s) {missing}; it has {sorted(available.values())}. "
                    f"Set SOI_FIELD_* (or fields=) to the right names.")
            cols = {k: available[v.lower()] for k, v in wanted.items()}
            for sr in reader.iterShapeRecords():
                if sr.shape.shapeType == shapefile.NULL:
                    continue
                rec = sr.record.as_dict()
                geom = shape(sr.shape.__geo_interface__)
                if not geom.is_valid:
                    geom = shapely.make_valid(geom)  # lookup copy only; source untouched
                if any(_missing(rec[cols[k]]) for k in cols):
                    skipped += 1  # keep the polygon so a hit is reported as incomplete, not "no polygon"
                    records.append(_IncompleteDistrict(tuple((cols[k], rec[cols[k]]) for k in cols)))
                    geoms.append(geom)
                    continue
                records.append(DistrictJurisdiction(
                    state_lgd=_code(rec[cols["state_lgd"]], cols["state_lgd"]),
                    district_lgd=_code(rec[cols["district_lgd"]], cols["district_lgd"]),
                    state_name=_name(rec[cols["state_name"]], cols["state_name"]),
                    district_name=_name(rec[cols["district_name"]], cols["district_name"]),
                ))
                geoms.append(geom)
        if skipped:
            warnings.warn(f"SOI shapefile {path}: {skipped} district record(s) have a "
                          f"missing/empty required attribute; points inside them are "
                          f"unresolvable (values are never filled in).")
        if not geoms:
            raise ValueError(f"SOI shapefile has no usable district polygons: {path}")
        self._records = records
        self._tree = STRtree(geoms)

    def resolve(self, latitude, longitude):
        """District/state of a WGS84 point; raises JurisdictionUnavailable."""
        from shapely.geometry import Point

        for name, v, lim in (("latitude", latitude, 90.0), ("longitude", longitude, 180.0)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) \
                    or abs(v) > lim:
                raise ValueError(f"{name} must be a finite WGS84 degree value within "
                                 f"[-{lim:g}, {lim:g}], got {v!r}")
        x, y = self._to_soi.transform(float(longitude), float(latitude))
        if not (math.isfinite(x) and math.isfinite(y)):
            raise JurisdictionUnavailable("point cannot be expressed in the SOI dataset's CRS")
        hits = self._tree.query(Point(x, y), predicate="intersects")
        if len(hits) == 0:
            raise JurisdictionUnavailable(
                f"no SOI district polygon contains ({latitude}, {longitude})")
        found = {self._records[i] for i in hits}
        if len(found) != 1:
            raise JurisdictionUnavailable(
                f"({latitude}, {longitude}) matches {len(found)} different SOI districts "
                f"(boundary or overlap); refusing to guess")
        only = found.pop()
        if isinstance(only, _IncompleteDistrict):
            gaps = [f for f, v in only.attrs if _missing(v)]
            raise JurisdictionDataIncomplete(
                f"({latitude}, {longitude}) is inside an SOI district polygon, but its required "
                f"attribute(s) {gaps} are missing; stored values: {dict(only.attrs)}. "
                f"Not inferring or mapping any code")
        return only


_cache = {}
_lock = threading.Lock()


def resolve_jurisdiction(latitude, longitude, *, shapefile_path=None, fields=None):
    """
    Resolve a WGS84 (latitude, longitude) to the SOI district and state.
    Returns a `DistrictJurisdiction` (state_lgd, district_lgd, state_name,
    district_name) or raises `JurisdictionUnavailable`. The dataset is loaded
    once per (path, fields) and cached.
    """
    path = shapefile_path or os.environ.get(SHAPEFILE_ENV)
    if not path:
        raise JurisdictionUnavailable(
            f"no SOI district shapefile configured (pass shapefile_path or set {SHAPEFILE_ENV})")
    key = (os.path.abspath(os.fspath(path)), tuple(sorted((fields or {}).items())))
    with _lock:
        resolver = _cache.get(key)
        if resolver is None:
            resolver = _cache[key] = SOIDistrictResolver(path, fields)
    return resolver.resolve(latitude, longitude)