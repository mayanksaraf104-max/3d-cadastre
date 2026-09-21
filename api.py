from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import shutil
import uuid
from typing import List, Optional

import config
# Import your master pipeline
import main

# Existing GPS -> jurisdiction resolver. The module file is named
# Jurisdiction_resolver.py; the lowercase fallback covers case-sensitive
# filesystems where the file was saved under a different casing.
try:
    from Jurisdiction_resolver import resolve_jurisdiction
except ImportError:
    try:
        from jurisdiction_resolver import resolve_jurisdiction
    except ImportError:
        resolve_jurisdiction = None
        print("⚠️  Jurisdiction_resolver.resolve_jurisdiction could not be imported -- "
              "automatic jurisdiction resolution is unavailable; requests must supply "
              "both state_code and district_code until it is.")

app = FastAPI(
    title="National True 3D Cadastre API",
    description="REST API for Automated 3D ULPIN Generation and Volumetric Property Mapping compliant with ISO 19152 (LADM).",
    version="3.1.0"
)

# CORS driven by CADASTRE_CORS_ORIGINS (config.py); an empty list (the
# default) means same-origin only, which is what you want if the
# frontend is served from this same API.
if config.ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    print("ℹ️  CADASTRE_CORS_ORIGINS not set -- cross-origin requests are disabled "
          "(same-origin only). Set it to a comma-separated list of allowed "
          "origins if the frontend is hosted separately from this API.")

if not config.API_KEY:
    print("⚠️  CADASTRE_API_KEY is not set -- registration/demolition endpoints "
          "are OPEN to anyone who can reach this API. Set CADASTRE_API_KEY "
          "before deploying anywhere reachable by the public.")


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """
    Intentionally minimal (a single shared key, not per-user accounts/roles) 
    because that's what's needed to stop this being wide open; swap it for real
    OAuth2/JWT + per-official accounts before this issues legally binding records.
    """
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
    return True

_FRONTEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")

if os.path.exists(_FRONTEND_PATH):
    print(f"🖥️  Frontend found: serving {_FRONTEND_PATH} at /app")
else:
    print(f"⚠️  Frontend NOT found at {_FRONTEND_PATH} -- /app will return a 500 until it exists.")

@app.get("/app")
def serve_frontend():
    if os.path.exists(_FRONTEND_PATH):
        return FileResponse(_FRONTEND_PATH, media_type="text/html")
    raise HTTPException(
        status_code=500,
        detail=f"Frontend file not found at {_FRONTEND_PATH}. "
               f"Make sure static/index.html sits directly next to api.py."
    )

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/tiff"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ALLOWED_LIDAR_EXTENSIONS = {".xyz", ".las", ".laz", ".ply", ".pts", ".txt", ".csv"}


def _looks_like_image(upload: UploadFile) -> bool:
    """
    Accepts if EITHER the declared content-type OR the filename
    extension looks like an image -- still rejects e.g. a .pdf or .exe,
    but doesn't punish a client for an unset header.
    """
    if upload.content_type in ALLOWED_IMAGE_TYPES:
        return True
    ext = os.path.splitext(upload.filename or "")[1].lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS


def _clean_code(value) -> Optional[str]:
    """Blank / whitespace-only / None -> None; anything else -> stripped string."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _code_to_str(value) -> Optional[str]:
    """
    Normalises a code returned by the resolver. Integer codes are zero-padded to
    the 2-digit form the ULPIN jurisdiction fields use (5 -> "05"); string codes
    are passed through untouched so the resolver stays authoritative.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"{value:02d}"
    return _clean_code(value)


def _normalize_resolved_jurisdiction(resolved):
    """
    Accepts the shapes a resolver commonly returns -- a dict, an object with
    attributes, or a (state, district) pair -- and returns (state, district)
    as strings, or None if either code is missing.
    """
    if resolved is None:
        return None

    state = district = None
    if isinstance(resolved, dict):
        state = resolved.get("state_code", resolved.get("state"))
        district = resolved.get("district_code", resolved.get("district"))
    elif isinstance(resolved, (tuple, list)):
        if len(resolved) >= 2:
            state, district = resolved[0], resolved[1]
    else:
        state = getattr(resolved, "state_code", getattr(resolved, "state_lgd", getattr(resolved, "state", None)))
        district = getattr(resolved, "district_code", getattr(resolved, "district_lgd", getattr(resolved, "district", None)))

    state, district = _code_to_str(state), _code_to_str(district)
    if not state or not district:
        return None
    return state, district


def _resolve_jurisdiction_codes(state_code, district_code, site_lat, site_lon, label: Optional[str] = None) -> dict:
    """
    Single source of truth for ULPIN jurisdiction on every registration path.

      * state_code AND district_code both supplied -> explicit manual override,
        used exactly as given (the resolver is not consulted).
      * both omitted/blank -> resolved automatically from site_lat/site_lon via
        Jurisdiction_resolver.resolve_jurisdiction().
      * only one supplied -> rejected; a half override is ambiguous.

    Nothing is hardcoded to any state or district.
    Returns {"state_code", "district_code", "source"} where source is
    "manual_override" or "auto_resolved".
    """
    prefix = f"{label}: " if label else ""
    state = _clean_code(state_code)
    district = _clean_code(district_code)

    if state and district:
        return {"state_code": state, "district_code": district, "source": "manual_override"}

    if state or district:
        raise HTTPException(
            status_code=400,
            detail=f"{prefix}state_code and district_code must be supplied together (manual override) "
                   f"or both omitted (jurisdiction is then resolved automatically from site_lat/site_lon)."
        )

    if site_lat is None or site_lon is None:
        raise HTTPException(
            status_code=400,
            detail=f"{prefix}cannot resolve jurisdiction automatically without site_lat and site_lon. "
                   f"Provide them, or supply both state_code and district_code as a manual override."
        )
    if not (-90.0 <= site_lat <= 90.0):
        raise HTTPException(status_code=400, detail=f"{prefix}site_lat {site_lat} is out of range (-90 to 90).")
    if not (-180.0 <= site_lon <= 180.0):
        raise HTTPException(status_code=400, detail=f"{prefix}site_lon {site_lon} is out of range (-180 to 180).")

    if resolve_jurisdiction is None:
        raise HTTPException(
            status_code=500,
            detail=f"{prefix}automatic jurisdiction resolution is unavailable on this server "
                   f"(Jurisdiction_resolver.resolve_jurisdiction could not be imported). "
                   f"Supply both state_code and district_code as a manual override."
        )

    try:
        resolved = resolve_jurisdiction(site_lat, site_lon)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"{prefix}jurisdiction could not be resolved from GPS ({site_lat}, {site_lon}): {e}"
        )

    codes = _normalize_resolved_jurisdiction(resolved)
    if codes is None:
        raise HTTPException(
            status_code=422,
            detail=f"{prefix}no jurisdiction found for GPS ({site_lat}, {site_lon}). "
                   f"The point may be outside covered boundaries; supply both state_code and "
                   f"district_code as a manual override."
        )

    print(f"🗺️  [API] {prefix}Jurisdiction auto-resolved from GPS ({site_lat}, {site_lon}): "
          f"state={codes[0]}, district={codes[1]}")
    return {"state_code": codes[0], "district_code": codes[1], "source": "auto_resolved"}


class InfrastructureProposal(BaseModel):
    project_name: str
    proposed_ewkt: str

@app.get("/")
def read_root():
    return {
        "status": "Online", 
        "system": "True 3D Cadastre Engine",
        "ogc_compliant": True,
        "standard": "ISO 19152 LADM Volume Supported"
    }

@app.post("/api/v1/process-cadastre")
async def process_cadastre(
    file: UploadFile = File(...),
    site_lat: Optional[float] = Form(None, description="Latitude from map click"),
    site_lon: Optional[float] = Form(None, description="Longitude from map click"),
    pixel_scale_m: Optional[float] = Form(None, description="Real-world meters represented by one blueprint pixel"),
    state_code: Optional[str] = Form(
        None, description="OPTIONAL 2-digit State Code. Supply together with district_code to override; "
                          "if both are omitted, jurisdiction is resolved automatically from site_lat/site_lon."
    ),
    district_code: Optional[str] = Form(
        None, description="OPTIONAL 2-digit District Code. Supply together with state_code to override; "
                          "if both are omitted, jurisdiction is resolved automatically from site_lat/site_lon."
    ),
    gcp_pixels: Optional[str] = Form(None, description="OPTIONAL JSON array of [[x,y],...] pixels"),
    gcp_real_world: Optional[str] = Form(None, description="OPTIONAL JSON array of [[x,y],...] UTM coords"),
    # floor_height and floor_count are OPTIONAL survey metadata / provenance ONLY.
    # They are validated and logged but are NEVER forwarded to the pipeline and can
    # never determine, position, trim, pair or construct geometry. Vertical geometry
    # is authoritative from measured 3D XYZ / B-Rep evidence; blueprint level marks,
    # massing codes and floor labels only provide semantic attribution/validation.
    floor_height: Optional[float] = Form(
        None, description="OPTIONAL survey metadata (metres), recorded for provenance only. Never drives geometry."
    ),
    floor_count: Optional[int] = Form(
        None, description="OPTIONAL survey metadata, recorded for provenance only. Never drives geometry."
    ),
    require_blueprint_evidence: bool = Form(
        True, description="Reject the upload when the drawing carries no recoverable vertical evidence. "
                          "Set False only for smoke tests -- a guessed storey count is not a legal record."
    ),
    is_demolition: bool = Form(False, description="Flag to authorize demolition of overlapping properties."),
    _auth: bool = Depends(require_api_key),
):
    # --- Input validation for True 3D Geographic Anchoring ---
    if not ((site_lat is not None and site_lon is not None and pixel_scale_m is not None) or 
            (gcp_pixels and gcp_real_world)):
        raise HTTPException(
            status_code=400, 
            detail="Missing georeferencing. Provide EITHER (site_lat, site_lon, pixel_scale_m) OR (gcp_pixels, gcp_real_world)."
        )

    if site_lat is not None and not (-90.0 <= site_lat <= 90.0):
        raise HTTPException(status_code=400, detail=f"site_lat {site_lat} is out of range (-90 to 90).")
    if site_lon is not None and not (-180.0 <= site_lon <= 180.0):
        raise HTTPException(status_code=400, detail=f"site_lon {site_lon} is out of range (-180 to 180).")
    
    if floor_count is not None and floor_count < 1:
        raise HTTPException(status_code=400, detail="floor_count survey metadata, if supplied, must be at least 1.")
    if floor_height is not None and floor_height <= 0:
        raise HTTPException(status_code=400, detail="floor_height survey metadata, if supplied, must be positive metres.")
    if not _looks_like_image(file):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file '{file.filename}' (content-type '{file.content_type}'). Upload a PNG, JPEG, or TIFF blueprint/aerial image."
        )

    # Jurisdiction: manual override if BOTH codes were supplied, otherwise
    # resolved from GPS. Done before any disk writes or pipeline work.
    jurisdiction = _resolve_jurisdiction_codes(state_code, district_code, site_lat, site_lon)
    state_code = jurisdiction["state_code"]
    district_code = jurisdiction["district_code"]

    # Safely parse GCP arrays if provided
    try:
        parsed_gcp_pixels = json.loads(gcp_pixels) if gcp_pixels else None
        parsed_gcp_real = json.loads(gcp_real_world) if gcp_real_world else None
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"GCP input is not valid JSON: {e}")

    file_ext = os.path.splitext(file.filename or "")[1] or ".png"
    temp_file_path = f"temp_upload_{uuid.uuid4().hex}{file_ext}"

    try:
        print(f"\n🌐 [API] Receiving 3D payload: {file.filename}")
        print(f"📍 [API] Location Set: Lat {site_lat}, Lon {site_lon} (Scale: {pixel_scale_m}m/px)")
        print(f"🗺️ [API] Jurisdiction ({jurisdiction['source']}): state={state_code}, district={district_code}")
        print(f"🏗️ [API] Demolish Existing Asset: {is_demolition}")
        print("🏢 [API] Vertical Profile: measured 3D XYZ/B-Rep evidence is authoritative")
        if floor_count is not None or floor_height is not None:
            print(f"⚠️ [API] Survey metadata (provenance only, NOT used for geometry): "
                  f"floors={floor_count}, floor_height={floor_height}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"🌐 [API] Triggering Master True 3D Pipeline...")
        
        result = main.run_unified_cadastre_pipeline(
            image_path=temp_file_path,
            site_lat=site_lat,
            site_lon=site_lon,
            state_code=state_code,
            district_code=district_code,
            pixel_scale_m=pixel_scale_m,
            gcp_pixels=parsed_gcp_pixels,
            gcp_real_world=parsed_gcp_real,
            floor_h=None,   # survey metadata is never forwarded as a geometry input
            floors=None,    # survey metadata is never forwarded as a geometry input
            is_demolition=is_demolition,
            require_blueprint_evidence=require_blueprint_evidence,
        ) or {}

        if result.get("error"):
            # A blueprint with no recoverable vertical evidence is a distinct,
            # ACTIONABLE failure -- the caller can fix it by uploading a proper sheet.
            if result.get("error_code") == "INSUFFICIENT_VERTICAL_EVIDENCE":
                return JSONResponse(
                    status_code=422,
                    content={
                        "status": "error",
                        "error_code": "INSUFFICIENT_VERTICAL_EVIDENCE",
                        "message": result["error"],
                        "remediation": {
                            "summary": "This drawing does not state the building's vertical structure.",
                            "accepted_evidence": [
                                "Level marks on the sheet, e.g. 'FFL +3.000' / 'EL. +6.00'",
                                "A massing code, e.g. 'G+3', 'S+4', or '2B+G+12'",
                                "Floor-panel titles, e.g. 'SECOND FLOOR PLAN'",
                                "Indexed LiDAR covering this footprint (see lidar_indexer.py)",
                            ],
                            "floor_metadata": "floor_count / floor_height are survey metadata only and "
                                              "cannot substitute for measured 3D XYZ/B-Rep evidence; "
                                              "blueprint labels only attribute/validate it.",
                        },
                    },
                )
            return JSONResponse(
                content={"status": "error", "message": result["error"]},
                status_code=422
            )

        if result.get("brep_count") == 0:
            # Nothing was reconstructed: later stages never ran, and the ledger GLB
            # on disk is NOT a model of this upload, so no model URL is offered.
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Reconstruction BLOCKED: 0 B-Reps reconstructed. "
                               "B-Rep validation, clash check and registration NOT RUN.",
                    "brep_count": 0,
                    "failed_units": result.get("failed_units", [])
                },
                status_code=422
            )

        if result.get("registered_count", 0) == 0:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Pipeline ran but no units were registered. Check server logs for clash/geometry rejections.",
                    "failed_units": result.get("failed_units", [])
                },
                status_code=422
            )

        model_ready = os.path.exists("approved_cadastre.glb")
        return JSONResponse(content={
            "status": "success",
            "message": "True 3D Pipeline execution fully complete. Volumetric assets registered in PostGIS.",
            "vertical_model": result.get("vertical_model"),
            "location": [site_lat, site_lon] if site_lat else "GCP_Anchored",
            "jurisdiction": jurisdiction,
            "registered_count": result.get("registered_count", 0),
            "ulpins": result.get("registered_units", []),
            "failed_units": result.get("failed_units", []),
            "model_download_url": "http://localhost:8000/api/v1/download-model" if model_ready else None
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def _associate_lidar_uploads(building_specs, lidar_files) -> dict:
    """
    Maps building index -> its LiDAR/XYZ UploadFile.

    The client sends only the LiDAR files that exist (not one per building), so
    they are matched by each building's `lidar_file_name`. Uploads are consumed
    first-come-first-served, so two buildings that reference the same filename
    each receive their own upload in request order. A referenced name with no
    upload, or an upload no building references, is rejected rather than
    silently dropped.
    """
    pool = list(lidar_files or [])
    assigned = {}
    for i, spec in enumerate(building_specs):
        if not isinstance(spec, dict):
            continue  # reported by the per-building validation loop
        name = spec.get("lidar_file_name")
        if name is None or name == "":
            continue
        label = spec.get("label") or f"building[{i}]"
        if not isinstance(name, str):
            raise HTTPException(status_code=400, detail=f"{label}: lidar_file_name must be a string.")
        wanted = os.path.basename(name.replace("\\", "/"))
        match = next(
            (u for u in pool if os.path.basename((u.filename or "").replace("\\", "/")) == wanted),
            None,
        )
        if match is None:
            raise HTTPException(
                status_code=400,
                detail=f"{label}: lidar_file_name '{name}' has no matching upload in `lidar_files`."
            )
        ext = os.path.splitext(match.filename or "")[1].lower()
        if ext not in ALLOWED_LIDAR_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"{label}: unsupported LiDAR/XYZ file '{match.filename}'. "
                       f"Accepted: {', '.join(sorted(ALLOWED_LIDAR_EXTENSIONS))}."
            )
        pool.remove(match)
        assigned[i] = match
    if pool:
        raise HTTPException(
            status_code=400,
            detail="`lidar_files` contains upload(s) no building references via lidar_file_name: "
                   + ", ".join(repr(u.filename) for u in pool)
        )
    return assigned


@app.post("/api/v1/process-cadastre-batch")
async def process_cadastre_batch(
    files: List[UploadFile] = File(..., description="One blueprint/aerial image per building, same order as `buildings`."),
    buildings: str = Form(..., description=(
        'JSON array, one object per file in `files`, same order. Each object: '
        '{"state_code": str (optional), "district_code": str (optional) -- supply BOTH to override, '
        'or omit both to auto-resolve jurisdiction from site_lat/site_lon, '
        '"site_lat": float (optional), "site_lon": float (optional), "pixel_scale_m": float (optional), '
        '"gcp_pixels": list (optional), "gcp_real_world": list (optional), '
        '"is_demolition": bool (optional), "label": str (optional), '
        '"lidar_file_name": str (optional) -- filename of this building\'s upload in `lidar_files`; omit if it has no LiDAR/XYZ evidence, '
        '"floor_height": float (OPTIONAL survey metadata, provenance only), '
        '"floor_count": int (OPTIONAL survey metadata, provenance only), '
        '"require_blueprint_evidence": bool (optional, default true)}. '
        "floor_height/floor_count are never forwarded and never drive geometry -- "
        "vertical geometry is authoritative from measured 3D XYZ/B-Rep evidence; "
        "blueprint level/floor labels only provide semantic attribution/validation."
    )),
    lidar_files: Optional[List[UploadFile]] = File(
        None, description="OPTIONAL LiDAR/XYZ evidence uploads. Each is matched to its building by that "
                          "building's `lidar_file_name`; only buildings that have LiDAR need an entry."
    ),
    _auth: bool = Depends(require_api_key),
):
    """
    Registers MULTIPLE buildings in one request.
    Vertical geometry is authoritative from measured 3D XYZ/B-Rep evidence only;
    floor_count/floor_height are optional survey metadata (provenance) and are
    never forwarded to the pipeline.
    """
    try:
        building_specs = json.loads(buildings)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"`buildings` is not valid JSON: {e}")

    if not isinstance(building_specs, list) or not building_specs:
        raise HTTPException(status_code=400, detail="`buildings` must be a non-empty JSON array.")
    if len(building_specs) != len(files):
        raise HTTPException(
            status_code=400,
            detail=f"Got {len(files)} file(s) but {len(building_specs)} building metadata "
                   f"object(s) -- these must be the same length and in the same order."
        )
    MAX_BATCH_SIZE = int(os.environ.get("CADASTRE_MAX_BATCH_SIZE", "25"))
    if len(building_specs) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch of {len(building_specs)} buildings exceeds the max of {MAX_BATCH_SIZE} "
                   f"per request. Split into smaller batches."
        )

    lidar_assignments = _associate_lidar_uploads(building_specs, lidar_files)

    # --- Validate every building's metadata up front, before touching disk ---
    resolved_jurisdictions = []
    for i, (spec, upload) in enumerate(zip(building_specs, files)):
        if not isinstance(spec, dict):
            raise HTTPException(status_code=400, detail=f"building[{i}]: metadata must be a JSON object.")
        label = spec.get("label") or f"building[{i}]"

        has_lat_lon = "site_lat" in spec and "site_lon" in spec and "pixel_scale_m" in spec
        has_gcp = "gcp_pixels" in spec and "gcp_real_world" in spec
        if not (has_lat_lon or has_gcp):
            raise HTTPException(
                status_code=400, 
                detail=f"{label}: missing georeferencing. Provide EITHER (site_lat, site_lon, pixel_scale_m) OR (gcp_pixels, gcp_real_world)."
            )

        try:
            if has_lat_lon:
                lat = float(spec["site_lat"]); lon = float(spec["site_lon"])
                if not (-90.0 <= lat <= 90.0):
                    raise HTTPException(status_code=400, detail=f"{label}: site_lat {lat} is out of range (-90 to 90).")
                if not (-180.0 <= lon <= 180.0):
                    raise HTTPException(status_code=400, detail=f"{label}: site_lon {lon} is out of range (-180 to 180).")
            
            fh = float(spec["floor_height"]) if spec.get("floor_height") is not None else None
            fc = int(spec["floor_count"]) if spec.get("floor_count") is not None else None
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{label}: coordinates and floor survey metadata must be numeric.")
        
        if fc is not None and fc < 1:
            raise HTTPException(status_code=400, detail=f"{label}: floor_count survey metadata, if supplied, must be at least 1.")
        if fh is not None and fh <= 0:
            raise HTTPException(status_code=400, detail=f"{label}: floor_height survey metadata, if supplied, must be positive metres.")
        if not _looks_like_image(upload):
            raise HTTPException(
                status_code=400,
                detail=f"{label}: unsupported file '{upload.filename}' (content-type '{upload.content_type}'). "
                       f"Upload a PNG, JPEG, or TIFF blueprint/aerial image."
            )

        # Jurisdiction per building: manual override if BOTH codes are present,
        # otherwise resolved from this building's own GPS coordinates.
        try:
            j_lat = float(spec["site_lat"]) if spec.get("site_lat") is not None else None
            j_lon = float(spec["site_lon"]) if spec.get("site_lon") is not None else None
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{label}: site_lat/site_lon must be numeric.")
        jurisdiction = _resolve_jurisdiction_codes(
            spec.get("state_code"), spec.get("district_code"), j_lat, j_lon, label=label
        )
        resolved_jurisdictions.append({"label": label, **jurisdiction})

    temp_paths = []
    try:
        jobs = []
        for i, (spec, upload) in enumerate(zip(building_specs, files)):
            file_ext = os.path.splitext(upload.filename or "")[1] or ".png"
            temp_path = f"temp_upload_{uuid.uuid4().hex}{file_ext}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)
            temp_paths.append(temp_path)

            lidar_path = None
            lidar_upload = lidar_assignments.get(i)
            if lidar_upload is not None:
                lidar_ext = os.path.splitext(lidar_upload.filename or "")[1].lower()
                lidar_path = f"temp_lidar_{uuid.uuid4().hex}{lidar_ext}"
                with open(lidar_path, "wb") as lidar_buffer:
                    shutil.copyfileobj(lidar_upload.file, lidar_buffer)
                temp_paths.append(lidar_path)

            jobs.append({
                "image_path": temp_path,
                "site_lat": float(spec["site_lat"]) if "site_lat" in spec else None,
                "site_lon": float(spec["site_lon"]) if "site_lon" in spec else None,
                "pixel_scale_m": float(spec["pixel_scale_m"]) if "pixel_scale_m" in spec else None,
                "state_code": resolved_jurisdictions[i]["state_code"],
                "district_code": resolved_jurisdictions[i]["district_code"],
                "gcp_pixels": spec.get("gcp_pixels"),
                "gcp_real_world": spec.get("gcp_real_world"),
                "floor_h": None,   # survey metadata is never forwarded as a geometry input
                "floors": None,    # survey metadata is never forwarded as a geometry input
                "require_blueprint_evidence": bool(spec.get("require_blueprint_evidence", True)),
                "is_demolition": bool(spec.get("is_demolition", False)),
                "label": spec.get("label") or f"{upload.filename or ('building_' + str(i + 1))}",
            })
            # Only present when LiDAR was supplied, so non-LiDAR jobs are byte-for-byte unchanged.
            if lidar_path:
                jobs[-1]["lidar_path"] = lidar_path

        print(f"\n🌐 [API] Batch registration: {len(jobs)} building(s)")
        batch_result = main.run_batch_cadastre_pipeline(jobs)

        # With 0 B-Reps reconstructed the GLB on disk is only the pre-existing ledger
        # export, not a model of this upload -- never offer it as the new model.
        model_ready = (os.path.exists("approved_cadastre.glb")
                       and batch_result.get("total_brep_count") != 0)
        return JSONResponse(content={
            "status": "success",
            "total_registered_count": batch_result["total_registered_count"],
            "total_brep_count": batch_result.get("total_brep_count"),
            "buildings": batch_result["buildings"],
            "jurisdictions": resolved_jurisdictions,
            "rejected_for_no_vertical_evidence": [
                b.get("label") for b in batch_result["buildings"]
                if b.get("error_code") == "INSUFFICIENT_VERTICAL_EVIDENCE"
            ],
            "model_download_url": "http://localhost:8000/api/v1/download-model" if model_ready else None,
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
    finally:
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)


@app.get("/api/v1/property/{ulpin}")
def get_property(ulpin: str):
    print(f"🔍 [API] Fetching legal record for ULPIN: {ulpin}")
    with main.CadastreDatabaseEngine() as db:
        record = db.get_property_record(ulpin)
    if record:
        return JSONResponse(content={"status": "success", "data": record})
    else:
        raise HTTPException(status_code=404, detail=f"ULPIN {ulpin} not found in the legal ledger.")

@app.post("/api/v1/audit-infrastructure")
def audit_infrastructure(proposal: InfrastructureProposal, _auth: bool = Depends(require_api_key)):
    print(f"🚧 [API] Running True 3D Spatial Audit for: {proposal.project_name}")
    with main.CadastreDatabaseEngine() as db:
        conflicts = db.audit_infrastructure_clash(proposal.proposed_ewkt)
    return JSONResponse(content={
        "status": "success",
        "project": proposal.project_name,
        "clash_count": len(conflicts),
        "affected_properties": conflicts,
        "message": "Audit complete. Affected properties must be cleared or compensated."
    })

@app.get("/api/v1/download-model")
def download_model():
    file_path = "approved_cadastre.glb"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="model/gltf-binary", filename="national_cadastre.glb")
    return JSONResponse(content={"status": "error", "message": "Model not found. Run pipeline first."}, status_code=404)