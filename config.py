"""
Centralized configuration for the True 3D Cadastre system.

FIX (production hardening): every module in this pipeline used to read
PG_DBNAME/PG_USER/PG_PASSWORD/PG_HOST itself, each with its own
hardcoded fallback (most defaulted the password to "mayank9431"). That
meant:
  1. A misconfigured deployment would silently fall back to a dev
     password/host instead of failing at startup.
  2. Every file had to remember the exact same four env var names and
     defaults, so a typo or drift in one file (see export_ledger.py's
     history) silently pointed it at a different database than the rest
     of the pipeline.

This module is now the single source of truth. Import `PG_DSN_KWARGS`,
`get_pool()`, `CADASTRE_SRID`, `ALLOWED_ORIGINS`, and `API_KEY` from
here instead of reading os.environ directly in new code.

Required env vars (no insecure defaults for credentials):
  PG_DBNAME, PG_USER, PG_PASSWORD, PG_HOST

Optional:
  PG_PORT                 (default 5432)
  CADASTRE_SRID            (default 7755 -- WGS 84 / India NSF LCC)
  CADASTRE_API_KEY         (if unset, mutating endpoints are OPEN -- see api.py)
  CADASTRE_CORS_ORIGINS     comma-separated allowed origins (default: none -- same-origin only)
  CADASTRE_DB_POOL_MIN/MAX  connection pool bounds (default 2 / 10)
  SLAB_CLASSIFICATION_CODE  LAS classification code for slab points (default 64)
  ROOF_CLASSIFICATION_CODE  LAS classification code for roof points (default 65)
  WALL_CLASSIFICATION_CODE  LAS classification code for wall points (default 66)
"""
import os
import sys
from pathlib import Path
from psycopg2 import pool as _pg_pool

_DOTENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_dotenv_manually(path: Path):
    """
    FIX: this used to depend on the third-party `python-dotenv` package.
    That turned into its own source of failure -- `pip install
    python-dotenv` silently not landing in the active venv, wrong
    `pip` on PATH, etc. -- which has nothing to do with the actual
    goal (get PG_DBNAME etc. into os.environ from a text file). A
    `.env` file is just `KEY=VALUE` lines; parsing that needs no
    dependency at all, so we do it inline instead of depending on
    an install step succeeding.

    Same semantics as python-dotenv's defaults: '#' starts a comment,
    blank lines are skipped, values may be wrapped in single/double
    quotes, and a real environment variable that's already set always
    wins over what's in the file.
    """
    if not path.exists():
        return False
    with open(path, "r", encoding="utf-8-sig") as f:  # utf-8-sig: tolerate a
        # BOM if the file was saved by Notepad, which likes to add one.
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    return True


if _load_dotenv_manually(_DOTENV_PATH):
    print(f"🔧 Loaded environment from {_DOTENV_PATH}")
else:
    print(f"ℹ️  No .env file at {_DOTENV_PATH} -- relying on real environment variables. "
          f"(Common Windows gotcha: Notepad may have saved it as '.env.txt' -- "
          f"check with `dir /a` in that folder.)")


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(
            f"❌ Missing required environment variable: {name}. "
            f"Refusing to start with an implicit/insecure default. "
            f"Set it (e.g. in a .env file or your process manager) and retry."
        )
    return val


# --- Database credentials: required, no fallback -----------------------
PG_DBNAME = _require_env("PG_DBNAME")
PG_USER = _require_env("PG_USER")
PG_PASSWORD = _require_env("PG_PASSWORD")
PG_HOST = _require_env("PG_HOST")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))

PG_DSN_KWARGS = dict(
    dbname=PG_DBNAME,
    user=PG_USER,
    password=PG_PASSWORD,
    host=PG_HOST,
    port=PG_PORT,
    connect_timeout=int(os.environ.get("CADASTRE_CONNECT_TIMEOUT_S", "5")),
)

# --- Spatial reference ---------------------------------------------------
CADASTRE_SRID = int(os.environ.get("CADASTRE_SRID", "7755"))

# --- LiDAR semantic classification codes -----------------------------------
# LAS classification values that identify structural element types. These
# are measured, per-point semantic provenance from the synthetic LAS
# benchmark only -- they are read as-is from the point cloud's
# `classification` dimension. Nothing here infers floor IDs, derives
# anything from Z/height, or generates geometry.
# Override via the environment variables of the same name.
def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        sys.exit(
            f"❌ Invalid integer for environment variable {name}: {raw!r}. "
            f"Expected a whole number (e.g. {default})."
        )


SLAB_CLASSIFICATION_CODE = _int_env("SLAB_CLASSIFICATION_CODE", 64)
ROOF_CLASSIFICATION_CODE = _int_env("ROOF_CLASSIFICATION_CODE", 65)
WALL_CLASSIFICATION_CODE = _int_env("WALL_CLASSIFICATION_CODE", 66)

# --- YOLO model paths ------------------------------------------------------
# Single source of truth for the model weights used across the pipeline.
# Defaults are the previously hardcoded paths; override via the environment
# variables of the same name. Nothing here moves, renames or selects files.
# Relative paths (defaults and overrides) are resolved against the project
# root (this file's directory), not the process working directory; an
# absolute override is kept as-is.
def _model_path(env_name: str, default: str) -> str:
    p = Path(os.environ.get(env_name, default))
    return str(p if p.is_absolute() else _DOTENV_PATH.parent / p)


YOLO_INFERENCE_MODEL_PATH = _model_path(
    "YOLO_INFERENCE_MODEL_PATH", "runs/segment/runs/sih_hybrid_model/weights/best.pt")
YOLO_TRAINING_CHECKPOINT_PATH = _model_path(
    "YOLO_TRAINING_CHECKPOINT_PATH", "runs/segment/runs/sih_gpu_model/weights/best.pt")

# --- API auth --------------------------------------------------------
# If unset, mutating endpoints stay open -- fine for a local demo, but
# api.py logs a loud warning at startup so this can't be an accident in
# a real deployment.
API_KEY = os.environ.get("CADASTRE_API_KEY")

# --- CORS ----------------------------------------------------------------
# FIX: was allow_origins=["*"] unconditionally in api.py. Default is now
# "no cross-origin access" unless explicitly configured, since "*" plus
# allow_credentials=True is also a combination most browsers/W3C spec
# treat as invalid anyway.
_origins_raw = os.environ.get("CADASTRE_CORS_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _origins_raw.split(",") if o.strip()]

# --- Shared connection pool -----------------------------------------------
# FIX: api.py previously opened a brand-new psycopg2 connection (via
# CadastreDatabaseEngine()) on every single request, and paid the
# setup_ladm_schema() DDL cost on every one of those, too. That doesn't
# survive concurrent load -- Postgres has a hard max_connections, and
# each connection is expensive to establish. A single process-wide pool,
# initialized once, is what makes this deployable.
_pool = None
_POOL_MIN = int(os.environ.get("CADASTRE_DB_POOL_MIN", "2"))
_POOL_MAX = int(os.environ.get("CADASTRE_DB_POOL_MAX", "10"))


def get_pool():
    global _pool
    if _pool is None:
        _pool = _pg_pool.ThreadedConnectionPool(_POOL_MIN, _POOL_MAX, **PG_DSN_KWARGS)
    return _pool