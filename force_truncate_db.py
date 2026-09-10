import psycopg2
import sys

DB_PARAMS = dict(
    dbname="cadastre_db",
    user="postgres",
    password="mayank9431",
    host="localhost",
    # FIX: bound the connection attempt itself, same reasoning as the
    # other scripts -- this already has a lock_timeout for the TRUNCATE
    # itself, but that only kicks in AFTER a connection exists.
    connect_timeout=5,
)

def force_truncate():
    conn = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        # autocommit=True means TRUNCATE never sits inside a transaction
        # waiting for ACCESS EXCLUSIVE — it fires immediately and locks
        # release the moment the statement finishes, not at COMMIT.
        conn.autocommit = True
        cur = conn.cursor()

        # Step 1: kill every other session on this DB so nothing is
        # holding a lock we have to wait for.
        cur.execute("""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = 'cadastre_db'
              AND pid <> pg_backend_pid()
              AND state IS DISTINCT FROM 'idle in transaction (aborted)';
        """)
        killed = cur.rowcount
        print(f"Killed {killed} stale session(s).")

        # Step 2: hard timeout — if we somehow still can't get the lock
        # in 5 seconds, fail loudly instead of hanging forever.
        cur.execute("SET lock_timeout = '5s';")

        # Step 3: truncate all cadastre tables in dependency order.
        cur.execute("""
            TRUNCATE
                property_spatial_shards,
                property_registry,
                ownership_rights,
                legal_deeds,
                parties
            CASCADE;
        """)
        print("Database truncated successfully!")
        cur.close()

    except psycopg2.errors.LockNotAvailable:
        print("ERROR: Could not acquire lock within 5 seconds.")
        print("A background PostgreSQL process (autovacuum?) is still holding it.")
        print("Wait 10 seconds and try again, or restart the PostgreSQL service:")
        print("  net stop postgresql-x64-18 && net start postgresql-x64-18")
        sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    force_truncate()