import psycopg2
import psycopg2.errors
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import config
from db_engine import CadastreDatabaseEngine

print("Connecting to PostgreSQL...")

# FIX: previously hardcoded user="postgres", password="mayank9431",
# host="localhost" -- a real deployment's Postgres superuser/admin
# credentials for provisioning the database are almost never the same
# as its app-level credentials, and definitely aren't a fixed hackathon
# password. This now requires PG_USER/PG_PASSWORD/PG_HOST/PG_DBNAME to
# be set explicitly (config.py fails fast if they're missing) -- point
# them at whatever role/host is allowed to CREATE DATABASE in your
# environment (often a separate admin account from the app's runtime
# PG_USER, in which case run this script once with the admin creds set).
_admin_kwargs = dict(config.PG_DSN_KWARGS)
target_dbname = _admin_kwargs.pop("dbname")

# 1. Connect to the default server to create our specific cadastre database
conn = psycopg2.connect(**_admin_kwargs)
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cursor = conn.cursor()

try:
    cursor.execute(f"CREATE DATABASE {psycopg2.extensions.quote_ident(target_dbname, conn)};")
    print(f"✅ Database '{target_dbname}' created successfully.")
except psycopg2.errors.DuplicateDatabase:
    print(f"ℹ️ Database '{target_dbname}' already exists -- continuing.")

cursor.close()
conn.close()

# 2. Reconnect directly to the new database to install the OGC features
print("Installing OGC Spatial Extensions...")
conn = psycopg2.connect(**config.PG_DSN_KWARGS)
cursor = conn.cursor()

# Enable the core PostGIS extension
cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")

# Enable the 3D volume and advanced curve math extension
cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis_sfcgal;")

conn.commit()
cursor.close()
conn.close()

# 3. Create the schema through the SINGLE authoritative definition,
#    CadastreDatabaseEngine.setup_ladm_schema() in db_engine.py (registry,
#    spatial shards, indexes, LADM tables). No table/index/SRID/geometry
#    definitions live in this script. Constructing the engine performs no
#    DDL; the strict runtime statement_timeout it sets is lifted for this
#    one-off session so index builds/migrations can finish.
print("Creating the 3D cadastre schema (db_engine.setup_ladm_schema)...")
with CadastreDatabaseEngine(use_pool=False) as db:
    db.cursor.execute("SET statement_timeout = 0;")
    db.conn.commit()
    db.setup_ladm_schema()

print("✅ True 3D Cadastre Database fully initialized and ready!")