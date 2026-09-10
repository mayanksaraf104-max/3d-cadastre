#!/usr/bin/env python3
"""
fetch_lidar.py
──────────────
Downloads sample urban LiDAR point-cloud data (.las / .laz) for the
3D Cadastre project.

Strategy (3-tier waterfall)
───────────────────────────
1. **OpenTopography REST API** – if an API key is provided.
   Sign up free at https://opentopography.org → MyOpenTopo → API Key.

2. **USGS TNM Access API** – queries The National Map to discover
   LiDAR .laz tiles for any bounding box. No API key needed.

3. **Direct USGS rockyweb download** – hardcoded known .laz tiles
   from the USGS 3DEP programme. No API key needed.

Output
──────
  data/raw_lidar/
  ├── <descriptive_name>.las   (or .laz)
  └── metadata.json            (provenance info)

Requirements
────────────
  pip install requests tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm


# ──────────────────────── defaults ─────────────────────────
OUTPUT_DIR = Path("data/raw_lidar")

# Downtown San Francisco – a dense urban area ideal for cadastre demos
DEFAULT_BBOX = {
    "south": 37.7870,
    "north": 37.7920,
    "west": -122.4050,
    "east": -122.3980,
}

# OpenTopography endpoint
OT_API_URL = "https://portal.opentopography.org/API/usgsdem"

# ── USGS TNM Access API (no auth needed) ──
TNM_API_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"

# ── Fallback: publicly hosted .laz samples ──
# Real USGS 3DEP tiles hosted on public infrastructure.
FALLBACK_URLS = [
    {
        "url": "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/USGS_LPC_CA_SanFrancisco_2016_LAS_2018/laz/USGS_LPC_CA_SanFrancisco_2016_10SEG6747_LAS_2018.laz",
        "filename": "san_francisco_3dep_sample.laz",
        "description": "USGS 3DEP LiDAR – San Francisco, CA (2016)",
    },
    {
        "url": "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/USGS_LPC_NY_NewYorkCity_2017_LAS_2020/laz/USGS_LPC_NY_NewYorkCity_2017_18TWL8356_LAS_2020.laz",
        "filename": "new_york_city_3dep_sample.laz",
        "description": "USGS 3DEP LiDAR – New York City (2017)",
    },
    {
        "url": "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/USGS_LPC_CO_Denver_2020_B20/laz/USGS_LPC_CO_Denver_2020_B20_13TDE5851.laz",
        "filename": "denver_3dep_sample.laz",
        "description": "USGS 3DEP LiDAR – Denver, CO (2020)",
    },
]

# Chunk size for streaming downloads
CHUNK_SIZE = 8192


# ──────────────── download helper ──────────────────────────
def _download_file(url: str, dest: Path, description: str = "") -> bool:
    """Stream-download a file with a progress bar. Returns True on success."""
    print(f"⬇️   Downloading: {description or url}")
    print(f"     → {dest}")

    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=f"  {dest.name}",
            ) as bar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    bar.update(len(chunk))

        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"     ✅  Saved ({size_mb:.1f} MB)")
        return True

    except requests.exceptions.HTTPError as exc:
        print(f"     ❌  HTTP error: {exc}")
        return False
    except requests.exceptions.ConnectionError:
        print("     ❌  Connection error – check your internet.")
        return False
    except requests.exceptions.Timeout:
        print("     ❌  Request timed out.")
        return False


# ─────────── method 1: OpenTopography API ──────────────────
def fetch_via_opentopography(
    api_key: str,
    bbox: dict,
    output_dir: Path,
) -> bool:
    """
    Request point-cloud data from the OpenTopography USGS 3DEP API.

    API docs: https://portal.opentopography.org/apidocs/
    """
    params = {
        "demtype": "USGS3DEP",       # or specific dataset ID
        "south":   bbox["south"],
        "north":   bbox["north"],
        "west":    bbox["west"],
        "east":    bbox["east"],
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }

    # Note: OpenTopography's point cloud API uses a different endpoint for
    # raw LAZ.  The /API/usgsdem endpoint returns raster DEMs.
    # For actual point clouds, the Global Data endpoint is:
    pc_url = "https://portal.opentopography.org/API/globaldem"
    params_pc = {
        "demtype":      "SRTMGL1",
        "south":        bbox["south"],
        "north":        bbox["north"],
        "west":         bbox["west"],
        "east":         bbox["east"],
        "outputFormat":  "GTiff",
        "API_Key":       api_key,
    }

    print("🌐  Attempting OpenTopography API request ...")
    print(f"     BBox: {bbox}")

    dest = output_dir / "opentopography_sample.tif"
    try:
        resp = requests.get(pc_url, params=params_pc, stream=True, timeout=120)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(dest, "wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, desc="  OT download"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                bar.update(len(chunk))

        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"     ✅  Saved ({size_mb:.1f} MB) → {dest}")
        return True

    except Exception as exc:
        print(f"     ⚠️  OpenTopography request failed: {exc}")
        return False


# ─────────── method 2: USGS TNM Access API ─────────────────
def fetch_via_tnm_api(
    bbox: dict,
    output_dir: Path,
    max_files: int = 1,
) -> bool:
    """
    Query the USGS TNM (The National Map) Access API to discover and
    download real LiDAR .laz tiles covering a bounding box.

    No API key required.
    Docs: https://tnmaccess.nationalmap.gov/api/v1/docs
    """
    print("\n🗺️   Querying USGS TNM Access API for LiDAR tiles ...")
    bbox_str = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
    print(f"     BBox: {bbox_str}")

    params = {
        "datasets": "Lidar Point Cloud (LPC)",
        "bbox": bbox_str,
        "prodFormats": "LAZ",
        "outputFormat": "JSON",
        "max": max_files,
    }

    try:
        resp = requests.get(TNM_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"     ⚠️  TNM API query failed: {exc}")
        return False

    items = data.get("items", [])
    total = data.get("total", 0)
    print(f"     Found {total} tile(s) in the TNM catalogue.")

    if not items:
        print("     No tiles available for this bounding box.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for item in items[:max_files]:
        download_url = item.get("downloadURL", "")
        title = item.get("title", "unknown")
        ext = Path(download_url).suffix or ".laz"
        safe_name = title.replace(" ", "_").replace("/", "_")[:80] + ext
        dest = output_dir / safe_name

        if dest.exists():
            print(f"     ⏭  Already exists: {dest.name}")
            success_count += 1
            continue

        ok = _download_file(download_url, dest, description=title)
        if ok:
            success_count += 1

    return success_count > 0


# ─────────── method 3: direct USGS 3DEP download ──────────
def fetch_via_usgs_direct(output_dir: Path, max_files: int = 1) -> bool:
    """
    Download publicly available .laz tiles directly from the USGS
    rockyweb delivery server. No API key required.
    """
    print("\n🗺️   Fetching LiDAR data from USGS 3DEP (direct download) ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for entry in FALLBACK_URLS[:max_files]:
        dest = output_dir / entry["filename"]
        if dest.exists():
            print(f"     ⏭  Already exists: {dest.name}")
            success_count += 1
            continue

        ok = _download_file(entry["url"], dest, description=entry["description"])
        if ok:
            success_count += 1
        else:
            print(f"     ⚠️  Failed to download {entry['filename']}, trying next ...")

    return success_count > 0


# ─────────── write provenance metadata ─────────────────────
def _write_metadata(output_dir: Path, source: str, details: dict) -> None:
    meta_path = output_dir / "metadata.json"
    meta = {
        "project":     "3D Cadastre – SIH 26011",
        "source":      source,
        "downloaded":  time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **details,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"📝  Metadata written to {meta_path}")


# ──────────────────────── main ─────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download sample urban LiDAR (.las/.laz) for the 3D Cadastre project.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-k", "--api-key",
        type=str,
        default=os.environ.get("OPENTOPOGRAPHY_API_KEY", ""),
        help="OpenTopography API key (or set OPENTOPOGRAPHY_API_KEY env var).",
    )
    parser.add_argument(
        "-n", "--max-files",
        type=int,
        default=1,
        help="Max number of .laz files to download (default: 1).",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help='Bounding box as JSON: \'{"south":37.78,"north":37.79,"west":-122.40,"east":-122.39}\'',
    )
    args = parser.parse_args()

    output_dir = args.output
    bbox = json.loads(args.bbox) if args.bbox else DEFAULT_BBOX

    print("=" * 60)
    print("  fetch_lidar.py – 3D Cadastre LiDAR Data Fetcher")
    print("=" * 60)

    success = False
    source_used = ""

    # ── Tier 1: OpenTopography API (if key provided) ──
    if args.api_key:
        success = fetch_via_opentopography(args.api_key, bbox, output_dir)
        if success:
            source_used = "OpenTopography API"
            _write_metadata(output_dir, source_used, {
                "endpoint": OT_API_URL,
                "bbox": bbox,
            })

    # ── Tier 2: USGS TNM Access API (no key needed, dynamic discovery) ──
    if not success:
        if not args.api_key:
            print("\n💡  No OpenTopography API key provided.")
            print("    Trying USGS TNM Access API (no key needed) ...\n")

        success = fetch_via_tnm_api(bbox, output_dir, max_files=args.max_files)
        if success:
            source_used = "USGS TNM Access API"
            _write_metadata(output_dir, source_used, {
                "endpoint": TNM_API_URL,
                "bbox": bbox,
            })

    # ── Tier 3: Direct USGS rockyweb download (hardcoded known tiles) ──
    if not success:
        print("\n⚠️  TNM API returned no results. Falling back to known USGS tiles ...")
        success = fetch_via_usgs_direct(output_dir, max_files=args.max_files)
        if success:
            source_used = "USGS 3DEP (direct)"
            files_downloaded = [
                e["filename"]
                for e in FALLBACK_URLS[:args.max_files]
                if (output_dir / e["filename"]).exists()
            ]
            _write_metadata(output_dir, source_used, {
                "files": files_downloaded,
                "note": "Public LiDAR tiles from USGS rockyweb delivery.",
            })

    if success:
        print(f"\n🎉  LiDAR data ready at: {output_dir.resolve()}")
        print(f"    Source: {source_used}")
        print("    Next steps:")
        print("    • Use laspy / pdal to inspect:  laspy info <file>.laz")
        print("    • Visualise in CloudCompare or pptk")
        print("    • Feed into your 3D geometry pipeline")
    else:
        print("\n❌  No data was downloaded. Check your network and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
