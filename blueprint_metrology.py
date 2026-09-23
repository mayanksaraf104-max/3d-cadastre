"""
blueprint_metrology.py
======================

Derives the VERTICAL structure of a building (how many storeys, where each
storey's floor sits, how tall each storey is) from evidence, instead of
asking the user for `floors=` and `floor_h=`.

WHY THIS MODULE EXISTS
----------------------
`main.py` used to take `floors` and `floor_h` as caller-supplied numbers and
then extrude the SAME footprint N times at a CONSTANT pitch. That is a 2.5D
model wearing a 3D costume: the Z axis carried no measured information, it
just replayed two numbers a human typed into a form.

WHAT IS AND IS NOT RECOVERABLE FROM A BLUEPRINT
-----------------------------------------------
Be clear-eyed about this, because it shapes the whole design:

  * A single floor-PLAN raster is one horizontal slice through a building.
    It does NOT contain the storey count. No amount of CV recovers a number
    that was never drawn. Anything claiming otherwise is guessing.

  * A real architectural blueprint SHEET, however, usually does carry the
    information, as *annotation* rather than as geometry:
      - sheet/panel titles: "GROUND FLOOR PLAN", "SECOND FLOOR PLAN",
        "TYPICAL FLOOR PLAN (2ND-5TH)"
      - massing shorthand, very common on Indian municipal drawings:
        "G+3", "B1+G+7", "2B+S+12", "P+4"
      - level marks: "FFL +3.000", "F.F.L. + 6.000", "EL. +9.00",
        "LVL +12.000"  -> consecutive differences ARE the floor-to-floor
        height, measured, not assumed
      - explicit notes: "FLOOR TO FLOOR HT. 3.15 M", "TYP. FLR HT 3000"

  * Independently, the LiDAR you already index in `lidar_indexer.py` /
    `z_engine.py` gives a MEASURED total building height (roof surface minus
    ground). Total height + measured floor pitch => storey count, with no
    human in the loop at all.

So this module reads the annotations (OCR), reads the LiDAR envelope, makes
them argue with each other, and reports what it can defend -- with a source
and a confidence attached to every number. When neither channel yields
evidence, it raises. That raise is the point: it is what forces the operator
to upload a blueprint that is actually a legal drawing rather than a sketch.

DEPENDENCIES
------------
OCR is optional at import time. If `pytesseract` / the Tesseract binary is
absent, the annotation channel degrades to "no evidence found" and the LiDAR
channel carries the result on its own. Nothing here hard-crashes on a missing
optional dep; it reports the degradation in `evidence`.
"""

import os
import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import pytesseract
    pytesseract.get_tesseract_version()  # raises if the Tesseract binary is missing / not on PATH
    _OCR_AVAILABLE = True
except Exception:  # pragma: no cover - missing binary raises more than ImportError
    pytesseract = None
    _OCR_AVAILABLE = False


# Anything outside this band is not a habitable storey pitch; it is an OCR
# misread (a dimension string, a door width, a drawing-scale number).
MIN_PLAUSIBLE_FLOOR_HEIGHT_M = 2.1
MAX_PLAUSIBLE_FLOOR_HEIGHT_M = 8.0

# DISABLED as evidence. An assumed pitch is not surveyed: it is no longer used
# to derive a storey count, a height, or any vertical value. Kept only so any
# external import of the name does not break.
ASSUMED_FLOOR_HEIGHT_M = 3.0


class BlueprintMetrologyError(Exception):
    """
    Raised when neither blueprint annotation nor LiDAR yields enough evidence
    to build a defensible vertical model. Callers should surface this to the
    operator as "this drawing is not survey-grade, upload a proper blueprint"
    rather than silently substituting a default storey count.
    """


@dataclass
class Storey:
    """One physical storey. `floor_level` follows the cadastre convention:
    0 = ground, positive = above ground, negative = basement."""
    floor_level: int
    # ANNOTATION METADATA ONLY -- TRUE-3D CONTRACT:
    # `base_offset` and `height` are provenance read off the drawing (level
    # marks / stated notes). They MUST NOT create, position, trim, pair,
    # extrude, or modify any B-Rep geometry. Z geometry comes exclusively from
    # measured evidence elsewhere in the pipeline.
    # `base_offset` is set ONLY when a level mark was actually read from the
    # sheet; it is None for storeys known only by count (never synthesised
    # from floor_level * height).
    base_offset: Optional[float]   # metres relative to ground datum, or None
    # Floor-to-floor height in metres, metadata only. Set ONLY from a measured
    # level-mark difference or an explicitly stated note; None when unmeasured
    # (e.g. the top storey, an invalid mark difference, or an assumed pitch).
    height: Optional[float]
    label: str = ""

    @property
    def tier(self) -> str:
        if self.floor_level < 0:
            return "SUBSURFACE"
        if self.floor_level == 0:
            return "SURFACE"
        return "AIR_RIGHTS"


@dataclass
class BuildingEnvelope:
    """
    Reports what the drawing/LiDAR SAY about the building, for attribution
    and cross-checking only.

    TRUE-3D CONTRACT: `storeys`, `base_offset`, `height`, `total_height`,
    `floors_above_ground` and `lidar_total_height` are annotations/counts/
    heights. They MUST NOT be used to create, position, trim, pair, extrude,
    or modify B-Rep geometry, and no floor Z position may be derived from
    them (or from total_height / floor_height, an assumed pitch, or a count).
    """
    storeys: List[Storey]
    floor_height_source: str = "unknown"     # "level_marks" | "note" | "assumed"
    storey_count_source: str = "unknown"     # "level_marks" | "floor_labels" | "massing_code" | "lidar"
    confidence: float = 0.0                  # 0.0 - 1.0
    evidence: List[str] = field(default_factory=list)
    lidar_total_height: Optional[float] = None

    @property
    def floors_above_ground(self) -> int:
        return sum(1 for s in self.storeys if s.floor_level >= 0)

    @property
    def basements(self) -> int:
        return sum(1 for s in self.storeys if s.floor_level < 0)

    @property
    def median_floor_height(self) -> float:
        # Metadata only. 0.0 means "no annotated height known", not zero height.
        known = [s.height for s in self.storeys if s.height is not None]
        return statistics.median(known) if known else 0.0

    @property
    def total_height(self) -> float:
        # Metadata only (see class contract). Non-zero ONLY when every storey
        # carries a base_offset read from a level mark; 0.0 means "not
        # annotated", never "zero height". Never synthesised from count*pitch.
        if not self.storeys or any(
            s.base_offset is None or s.height is None for s in self.storeys
        ):
            return 0.0
        top = max(s.base_offset + s.height for s in self.storeys)
        bottom = min(s.base_offset for s in self.storeys)
        return top - bottom

    def describe(self) -> str:
        return (
            f"{self.floors_above_ground} storey(s) above ground"
            f"{f' + {self.basements} basement(s)' if self.basements else ''}, "
            f"median pitch "
            f"{f'{self.median_floor_height:.2f}m' if self.median_floor_height else 'n/a'} "
            f"[count:{self.storey_count_source}, pitch:{self.floor_height_source}, "
            f"confidence {self.confidence:.0%}]"
        )


# ======================================================================
# 1. OCR LAYER
# ======================================================================

_ORDINAL_WORDS = {
    "GROUND": 0, "GRND": 0, "GF": 0, "G.F": 0, "LOWER GROUND": -1, "LG": -1,
    "FIRST": 1, "1ST": 1, "SECOND": 2, "2ND": 2, "THIRD": 3, "3RD": 3,
    "FOURTH": 4, "4TH": 4, "FIFTH": 5, "5TH": 5, "SIXTH": 6, "6TH": 6,
    "SEVENTH": 7, "7TH": 7, "EIGHTH": 8, "8TH": 8, "NINTH": 9, "9TH": 9,
    "TENTH": 10, "10TH": 10, "ELEVENTH": 11, "11TH": 11, "TWELFTH": 12, "12TH": 12,
}


def ocr_text(image_path: str) -> str:
    """
    Returns the OCR'd text of the sheet, upper-cased, with runs of whitespace
    collapsed. Returns "" (never raises) if OCR is unavailable or fails --
    the caller treats that as "annotation channel produced no evidence".

    Blueprint text is small and thin, so we upscale and binarise first;
    running Tesseract on a raw 1:1 architectural scan reads almost nothing.
    """
    if not _OCR_AVAILABLE or cv2 is None:
        return ""
    if not os.path.exists(image_path):
        return ""

    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""

        # Upscale small scans so 8-10px annotation text reaches a size
        # Tesseract can actually resolve.
        h, w = img.shape[:2]
        if max(h, w) < 2200:
            scale = 2200.0 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        img = cv2.bilateralFilter(img, 5, 50, 50)
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )

        # PSM 11 = sparse text. A blueprint is exactly that: isolated labels
        # scattered over line-work, not paragraphs.
        raw = pytesseract.image_to_string(img, config="--psm 11")
        return re.sub(r"[ \t]+", " ", raw.upper())
    except Exception as e:
        print(f"   ⚠️ OCR pass failed ({e}); falling back to LiDAR-only metrology.")
        return ""


# ======================================================================
# 2. ANNOTATION PARSERS
# ======================================================================

def parse_level_marks(text: str) -> List[float]:
    """
    Extracts finished-floor-level marks and returns them in METRES, sorted
    and de-duplicated.

    Matches the forms that actually appear on drawings:
        FFL +3.000    F.F.L. +3.000    SFL +6.00
        EL. +9.000    LVL +12.000      RL +15.00
        + 3.000 M     +3000            (bare, mm)

    Units: values with |v| >= 50 are read as millimetres (a 3000 mm level
    mark), otherwise as metres. No real building has a 50 m floor-to-floor
    pitch, and no drawing writes a level as 3 mm, so the split is safe.
    """
    pattern = re.compile(
        r"(?<![A-Z0-9])(?:F\.?F\.?L|S\.?F\.?L|F\.?L|E\.?L|L\.?V\.?L|R\.?L|LEVEL)\s*[.:]?\s*"
        r"([+-]?\s*\d+(?:[.,]\d+)?)\s*(MM|M)?\b"
    )

    levels = []
    for raw_val, unit in pattern.findall(text):
        # A bare number after the keyword is not an architectural level mark.
        if not unit and raw_val.strip()[:1] not in ("+", "-"):
            continue
        try:
            val = float(raw_val.replace(" ", "").replace(",", "."))
        except ValueError:
            continue

        if unit == "MM" or (unit != "M" and abs(val) >= 50):
            val /= 1000.0

        # Sanity band: a level mark on a building sheet lives roughly
        # between 4 basements down and a 60-storey tower up.
        if -30.0 <= val <= 250.0:
            levels.append(round(val, 3))

    # Collapse near-duplicates (the same level annotated on several panels).
    levels = sorted(set(levels))
    collapsed: List[float] = []
    for lv in levels:
        if not collapsed or abs(lv - collapsed[-1]) > 0.25:
            collapsed.append(lv)
    return collapsed


def parse_massing_code(text: str) -> Optional[Tuple[int, int]]:
    """
    Parses Indian-style massing shorthand into (floors_above_ground, basements).

        "G+3"        -> (4, 0)   ground + 3 upper = 4 storeys above ground
        "2B+G+12"    -> (13, 2)
        "S+4"        -> (5, 0)   stilt + 4 upper
        "B+P+5"      -> (6, 1)   basement + parking + 5 upper

    This is the single highest-value token on a municipal sanction drawing:
    it states the storey count explicitly and unambiguously.
    """
    # Look for Ground (G), Stilt (S), or Parking (P) as the base level
    m = re.search(r"(?:(\d{0,2})\s*B\s*\+\s*)?[GSP]\s*\+\s*(\d{1,2})\b", text)
    if not m:
        return None

    basement_token, upper_token = m.groups()
    try:
        upper = int(upper_token)
    except (TypeError, ValueError):
        return None

    if basement_token is None:
        basements = 0
    elif basement_token == "":
        basements = 1          # bare "B+G+7"
    else:
        basements = int(basement_token)

    if not (0 <= upper <= 60) or not (0 <= basements <= 6):
        return None

    return upper + 1, basements   # +1 for the base (Ground/Stilt/Parking) floor itself


def parse_floor_labels(text: str) -> List[int]:
    """
    Finds panel/sheet titles naming a storey and returns the distinct floor
    levels mentioned (0 = ground, -1 = basement, etc).

    Handles "GROUND FLOOR PLAN", "2ND FLOOR PLAN", "LEVEL 5 PLAN",
    "BASEMENT 2 PLAN", and typical-floor ranges like
    "TYPICAL FLOOR PLAN (2ND - 5TH)" which expand to 2,3,4,5.
    """
    found = set()

    # Typical-floor ranges first: they imply storeys never drawn individually.
    for lo, hi in re.findall(
        r"TYP(?:ICAL)?[^.\n]{0,30}?(\d{1,2})\s*(?:ST|ND|RD|TH)?\s*(?:-|TO|–)\s*(\d{1,2})\s*(?:ST|ND|RD|TH)",
        text,
    ):
        try:
            lo_i, hi_i = int(lo), int(hi)
        except ValueError:
            continue
        if 0 <= lo_i <= hi_i <= 60:
            found.update(range(lo_i, hi_i + 1))

    for m in re.finditer(r"BASEMENT\s*(\d)?", text):
        depth = int(m.group(1)) if m.group(1) else 1
        if 1 <= depth <= 6:
            found.add(-depth)

    for m in re.finditer(r"(?:LEVEL|FLOOR)\s*[-:]?\s*(\d{1,2})\b", text):
        lvl = int(m.group(1))
        if 0 <= lvl <= 60:
            found.add(lvl)

    for word, lvl in _ORDINAL_WORDS.items():
        # Require the word to sit next to "FLOOR"/"PLAN" so a stray "FIRST"
        # in a general note does not invent a storey.
        if re.search(rf"\b{re.escape(word)}\b[^A-Z0-9]{{0,12}}(FLOOR|FLR|PLAN)", text):
            found.add(lvl)

    return sorted(found)


def parse_stated_floor_height(text: str) -> Optional[float]:
    """
    Extracts an explicitly stated floor-to-floor / storey height in metres.

        "FLOOR TO FLOOR HT. 3.15 M"
        "TYP. FLOOR HEIGHT 3000"
        "STOREY HEIGHT : 3.30M"
    """
    pattern = re.compile(
        r"(?:FLOOR\s*(?:TO\s*FLOOR)?|FLR|STOREY|STORY)\s*"
        r"(?:HT|HGT|HEIGHT)\s*[.:=]?\s*(\d+(?:[.,]\d+)?)\s*(MM|M)?\b"
    )
    candidates = []
    for raw_val, unit in pattern.findall(text):
        try:
            val = float(raw_val.replace(",", "."))
        except ValueError:
            continue
        if unit == "MM" or (unit != "M" and val >= 50):
            val /= 1000.0
        if MIN_PLAUSIBLE_FLOOR_HEIGHT_M <= val <= MAX_PLAUSIBLE_FLOOR_HEIGHT_M:
            candidates.append(val)

    return round(statistics.median(candidates), 3) if candidates else None


# ======================================================================
# 3. LIDAR CHANNEL
# ======================================================================

def lidar_building_height(footprint_global, z_dem, roof_model, roof_features,
                          z_roof=None) -> Optional[float]:
    """
    Reports total building height from MEASURED LiDAR evidence, as metadata
    for attribution/cross-checking only.

    The height is `z_roof - z_dem`, where `z_roof` comes straight from the
    measured point cloud (z_engine's 95th percentile of raw returns). It is
    never computed from a fitted surface: there is NO polynomial Z = f(X, Y)
    path here, and `roof_model` / `roof_features` / `footprint_global` are
    accepted only for call-site compatibility and are ignored.

    TRUE-3D CONTRACT: the returned value MUST NOT create, position, trim,
    pair, extrude, or modify B-Rep geometry.

    Returns None when no measured `z_roof` is available (no LiDAR coverage).
    """
    if z_roof is not None and z_roof != 0.0:
        height = float(z_roof) - float(z_dem)
        return height if height > 0 else None
    return None


# ======================================================================
# 4. RESOLVER
# ======================================================================

def _floor_height_from_levels(levels: List[float]) -> Optional[float]:
    """Median of consecutive differences between level marks, filtered to
    plausible storey pitches. This is the ONLY fully measured pitch we can
    get from the sheet, so it outranks stated notes."""
    if len(levels) < 2:
        return None
    diffs = [
        b - a for a, b in zip(levels, levels[1:])
        if MIN_PLAUSIBLE_FLOOR_HEIGHT_M <= (b - a) <= MAX_PLAUSIBLE_FLOOR_HEIGHT_M
    ]
    return round(statistics.median(diffs), 3) if diffs else None


def _storeys_from_levels(levels: List[float], fallback_height: Optional[float] = None) -> List[Storey]:
    """Builds storey metadata directly from measured level marks: each mark is
    a floor slab, and each storey's height is the distance to the NEXT mark --
    so a double-height ground floor or a taller podium survives instead of
    being flattened to a constant pitch.

    Only ACTUAL measured differences are kept. If the difference to the next
    mark is implausible, or there is no next mark (top storey), `height` is
    None -- it is never replaced by a median, stated, or assumed pitch.
    `fallback_height` is accepted for API compatibility and is ignored.

    `base_offset` / `height` here are annotation metadata read from the sheet
    (attribution only); they MUST NOT drive B-Rep geometry."""
    storeys: List[Storey] = []
    ground_idx = min(range(len(levels)), key=lambda i: abs(levels[i]))

    for i, base in enumerate(levels):
        height: Optional[float] = None
        if i + 1 < len(levels):
            diff = levels[i + 1] - base
            if MIN_PLAUSIBLE_FLOOR_HEIGHT_M <= diff <= MAX_PLAUSIBLE_FLOOR_HEIGHT_M:
                height = round(diff, 3)

        storeys.append(Storey(
            floor_level=i - ground_idx,
            base_offset=round(base - levels[ground_idx], 3),
            height=height,
            label=f"Level {i - ground_idx}",
        ))
    return storeys


def _storeys_from_count(above: int, basements: int,
                        floor_height: Optional[float] = None) -> List[Storey]:
    """Count-only storey records, used when we know the COUNT but no per-level Z.

    TRUE-3D CONTRACT: this does NOT build a Z stack. `base_offset` is always
    None -- it is never synthesised as floor_level * floor_height -- and
    `height` is `floor_height` as given (pass None when the pitch is assumed
    rather than measured/stated), carried as provenance only. It can never
    return Z coordinates. Nothing returned here may create, position, trim, pair, extrude, or modify B-Rep
    geometry.
    """
    storeys = []
    for b in range(basements, 0, -1):
        storeys.append(Storey(
            floor_level=-b,
            base_offset=None,
            height=floor_height,
            label=f"Basement {b}",
        ))
    for f in range(above):
        storeys.append(Storey(
            floor_level=f,
            base_offset=None,
            height=floor_height,
            label="Ground" if f == 0 else f"Floor {f}",
        ))
    return storeys


# Public alias kept for API compatibility (main.py's surveyor-override path).
# It yields count-only metadata records with base_offset=None; it is NOT a
# geometry-capable Z stack and must not be used to position floors.
build_uniform_storeys = _storeys_from_count


def resolve_building_envelope(
    image_path: str,
    footprint_global=None,
    z_dem: float = 0.0,
    z_roof: float = None,
    roof_model=None,
    roof_features=None,
    require_evidence: bool = True,
) -> BuildingEnvelope:
    """
    The single entry point `main.py` calls in place of trusting `floors=` and
    `floor_h=` from a form.

    Resolution order for the PITCH (floor-to-floor height):
        1. differences between measured level marks   (measured, best)
        2. an explicitly stated floor-height note     (stated)
        3. none -- no pitch is assumed (unavailable)

    Resolution order for the STOREY COUNT:
        1. level marks       - one slab per mark, per-storey heights preserved
        2. massing code      - "G+3" or "S+4", explicit and unambiguous
        3. LiDAR             - measured total height / measured-or-stated pitch
        (floor labels are OCR title matches only: reported, never proof of a count)

    `require_evidence=True` (the default) raises BlueprintMetrologyError when
    every channel comes up empty, rather than quietly defaulting to a
    single-storey box. Callers should let that error reach the operator.
    """
    evidence: List[str] = []

    text = ocr_text(image_path)
    if not text:
        evidence.append(
            "No OCR text recovered from the sheet"
            + ("" if _OCR_AVAILABLE else " (pytesseract or the Tesseract binary is not available)")
        )

    levels = parse_level_marks(text)
    massing = parse_massing_code(text)
    labels = parse_floor_labels(text)
    stated_height = parse_stated_floor_height(text)
    measured_height = _floor_height_from_levels(levels)

    if levels and measured_height is None:
        evidence.append(
            f"Level marks {levels} rejected: no adjacent pair forms a plausible storey pitch "
            f"({MIN_PLAUSIBLE_FLOOR_HEIGHT_M}-{MAX_PLAUSIBLE_FLOOR_HEIGHT_M}m); "
            f"not treated as level annotations."
        )
        levels = []
    if levels:
        evidence.append(f"Level marks read from sheet: {levels}")
    if massing:
        evidence.append(f"Massing code parsed: {massing[0]} above ground, {massing[1]} basement(s)")
    if labels:
        evidence.append(f"Floor labels found: {labels} (OCR title matches only; NOT proof of storey count)")
    if stated_height:
        evidence.append(f"Stated floor height note: {stated_height}m")

    lidar_h = lidar_building_height(footprint_global, z_dem, roof_model, roof_features,
                                    z_roof=z_roof)
    if lidar_h:
        evidence.append(f"LiDAR-measured total height: {lidar_h:.2f}m")

    # ---- pitch ----
    if measured_height:
        floor_height, height_source, height_conf = measured_height, "level_marks", 1.0
    elif stated_height:
        floor_height, height_source, height_conf = stated_height, "note", 0.8
    else:
        floor_height, height_source, height_conf = None, "unavailable", 0.3
        evidence.append(
            "⚠️ No measured or stated floor height on the sheet; no pitch is assumed."
        )

    # Pitch recorded on count-only storeys: measured/stated only (None when unavailable).
    meta_height = floor_height

    # ---- storey count ----
    if len(levels) >= 2:
        storeys = _storeys_from_levels(levels)
        count_source, count_conf = "level_marks", 1.0

    elif massing:
        storeys = _storeys_from_count(massing[0], massing[1], meta_height)
        count_source, count_conf = "massing_code", 0.95

    elif lidar_h:
        # Storey-COUNT ESTIMATE only (metadata). This ratio yields a number,
        # never a floor Z position; it must not drive B-Rep geometry. When the
        # pitch is assumed, the estimate is flagged low-confidence below.
        if floor_height:
            above = max(1, int(round(lidar_h / floor_height)))
            storeys = _storeys_from_count(above, 0, meta_height)
            count_source, count_conf = "lidar", 0.6
            evidence.append(
                f"Storey count derived from LiDAR envelope: "
                f"{lidar_h:.2f}m / {floor_height:.2f}m -> {above} storey(s)"
            )
        else:
            storeys = []
            count_source, count_conf = "lidar_height_only", 0.6
            evidence.append(
                f"LiDAR-measured total height {lidar_h:.2f}m established; storey count NOT "
                f"established (no measured or stated floor height, none assumed)."
            )

    else:
        if require_evidence:
            raise BlueprintMetrologyError(
                "Cannot determine the building's vertical structure from this upload.\n"
                "None of the following were found:\n"
                "  • level marks (e.g. 'FFL +3.000') on the drawing\n"
                "  • a massing code (e.g. 'G+3' or 'S+4') on the drawing\n"
                "  • LiDAR coverage over this footprint to measure total height\n"
                "Upload a blueprint sheet that carries floor annotations, or index "
                "LiDAR covering this site (see lidar_indexer.py). Storey count will "
                "NOT be guessed -- a guessed Z axis is not a cadastral record. "
                "Floor-label OCR matches, bare OCR numbers and assumed floor heights "
                "are not accepted as evidence.\n"
                f"Diagnostics: OCR available={_OCR_AVAILABLE}, OCR characters read={len(text)}, "
                f"LiDAR-measured height={lidar_h}."
            )
        storeys = _storeys_from_count(1, 0, meta_height)
        count_source, count_conf = "assumed_single_storey", 0.1
        evidence.append("⚠️ No vertical evidence at all; falling back to a single storey.")

    # ---- cross-check the two independent channels against each other ----
    # Compared only when EVERY above-ground height is known; a partial sum
    # (e.g. level marks with an unmeasured top storey) would be misleading.
    above_h = [s.height for s in storeys if s.floor_level >= 0]
    envelope_h = sum(above_h) if above_h and None not in above_h else 0.0
    if lidar_h and count_source != "lidar" and envelope_h > 0:
        discrepancy = abs(lidar_h - envelope_h) / max(lidar_h, envelope_h)
        if discrepancy > 0.25:
            count_conf *= 0.7
            evidence.append(
                f"⚠️ Annotation-derived height ({envelope_h:.1f}m) disagrees with "
                f"LiDAR ({lidar_h:.1f}m) by {discrepancy:.0%}. Both retained; "
                f"annotations kept as the legal source, confidence reduced."
            )
        else:
            count_conf = min(1.0, count_conf * 1.1)
            evidence.append(
                f"✅ Annotation height ({envelope_h:.1f}m) corroborated by LiDAR "
                f"({lidar_h:.1f}m), within {discrepancy:.0%}."
            )

    return BuildingEnvelope(
        storeys=storeys,
        floor_height_source=height_source,
        storey_count_source=count_source,
        confidence=round(min(count_conf, 1.0) * height_conf, 3),
        evidence=evidence,
        lidar_total_height=lidar_h,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python blueprint_metrology.py <blueprint_image>")
        sys.exit(1)

    try:
        env = resolve_building_envelope(sys.argv[1], require_evidence=True)
    except BlueprintMetrologyError as e:
        print(f"❌ {e}")
        sys.exit(2)

    print(f"\n🏢 {env.describe()}\n")
    for s in env.storeys:
        base = f"{s.base_offset:+.2f}m" if s.base_offset is not None else "n/a"
        hgt = f"{s.height:.2f}m" if s.height is not None else "n/a"
        print(f"   [{s.floor_level:+d}] {s.label:<12} base {base}  "
              f"height {hgt}  tier={s.tier}")
    print("\n📋 Evidence:")
    for line in env.evidence:
        print(f"   - {line}")