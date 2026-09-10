import psycopg2

# FIX: without connect_timeout, if the server is up but not yet accepting
# connections (e.g. mid crash-recovery right after a restart) or otherwise
# unreachable in a non-instant way, this connect() call can hang with no
# bound at all -- looks identical to a frozen terminal, but with nothing to
# Ctrl+C into yet. 5s is generous for localhost.
conn = psycopg2.connect(dbname="cadastre_db", user="postgres", password="mayank9431", host="localhost", connect_timeout=5)
cur = conn.cursor()
cur.execute("""
    SELECT pid, state, query, now() - xact_start AS xact_age
    FROM pg_stat_activity
    WHERE datname = 'cadastre_db' AND pid <> pg_backend_pid();
""")
rows = cur.fetchall()

if not rows:
    print("No other sessions connected to cadastre_db.")
else:
    print(f"Found {len(rows)} other session(s):\n")
    for pid, state, query, xact_age in rows:
        print(f"pid={pid}  state={state}  xact_age={xact_age}")
        print(f"  query: {query}")
        print()

cur.close()
conn.close()