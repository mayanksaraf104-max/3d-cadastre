import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

print("Connecting to PostgreSQL...")

# 1. Connect to the default server to create our specific cadastre database
conn = psycopg2.connect(user="postgres", password="mayank9431", host="localhost")
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cursor = conn.cursor()

try:
    cursor.execute("CREATE DATABASE cadastre_db;")
    print("✅ Database 'cadastre_db' created successfully.")
except Exception as e:
    print("⚠️ Database 'cadastre_db' already exists.")

cursor.close()
conn.close()

# 2. Reconnect directly to the new 'cadastre_db' to install the OGC features
print("Installing OGC Spatial Extensions...")
conn = psycopg2.connect(dbname="cadastre_db", user="postgres", password="mayank9431", host="localhost")
cursor = conn.cursor()

# Enable the core PostGIS extension
cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")

# Enable the 3D volume and advanced curve math extension
cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis_sfcgal;")

# 3. Create the official registry ledger (Master Legal Record)
print("Creating the Master 3D Property Registry table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS property_registry (
        ulpin VARCHAR(64) PRIMARY KEY,
        unit_id VARCHAR(50) NOT NULL,
        boundary geometry(GeometryZ, 4326) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

# 4. Create the Spatial Shard table for R-Tree micro-binning
print("Creating the Spatial Shards table for optimization...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS property_spatial_shards (
        id SERIAL PRIMARY KEY,
        ulpin VARCHAR(64) REFERENCES property_registry(ulpin) ON DELETE CASCADE,
        shard_geom geometry(GeometryZ, 4326) NOT NULL
    );
""")

# 5. Create the 3D bounding-box index ON THE SHARDS for lightning-fast clash detection
print("Building the N-Dimensional GiST Index on the spatial shards...")
cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_shards_3d 
    ON property_spatial_shards USING GIST (shard_geom gist_geometry_ops_nd);
""")

# 6. LADM tables (parties, deeds, ownership) — created here once,
#    not on every upload inside CadastreDatabaseEngine.__init__.
print("Creating LADM legal tables...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS parties (
        party_id SERIAL PRIMARY KEY,
        full_name VARCHAR(255) NOT NULL,
        national_id VARCHAR(50) UNIQUE NOT NULL,
        party_type VARCHAR(50)
    );
    CREATE TABLE IF NOT EXISTS legal_deeds (
        deed_id SERIAL PRIMARY KEY,
        deed_number VARCHAR(100) UNIQUE NOT NULL,
        issue_date DATE NOT NULL,
        encumbrance_status VARCHAR(100) DEFAULT 'CLEAR'
    );
    CREATE TABLE IF NOT EXISTS ownership_rights (
        right_id SERIAL PRIMARY KEY,
        ulpin VARCHAR(64) REFERENCES property_registry(ulpin) ON DELETE CASCADE,
        party_id INT REFERENCES parties(party_id) ON DELETE CASCADE,
        deed_id INT REFERENCES legal_deeds(deed_id) ON DELETE CASCADE,
        right_type VARCHAR(50),
        fractional_share DECIMAL(5,4) DEFAULT 1.0000
    );
""")

conn.commit()
print("✅ True 3D Cadastre Database fully initialized and ready!")

cursor.close()
conn.close()