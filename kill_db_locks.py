import psycopg2

# FIX: same reasoning as check_db_locks.py -- bound how long a connection
# attempt itself can hang before we even get to the query.
conn = psycopg2.connect(dbname="cadastre_db", user="postgres", password="mayank9431", host="localhost", connect_timeout=5)
conn.autocommit = True
cur = conn.cursor()
cur.execute("""
    SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE datname = 'cadastre_db' AND pid <> pg_backend_pid();
""")
results = cur.fetchall()
print(f"Terminated {len(results)} session(s).")

cur.close()
conn.close()