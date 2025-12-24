import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
conn.autocommit = True  # VACUUM requires autocommit
cur = conn.cursor()

print("Checking current chunk count...")
cur.execute("SELECT COUNT(*) FROM chunks")
print(f"  Chunks: {cur.fetchone()[0]:,}")

cur.execute("SELECT book_name, COUNT(*) FROM chunks GROUP BY book_name")
for row in cur.fetchall():
    print(f"    {row[0]}: {row[1]:,}")

print("\nRunning VACUUM FULL...")
cur.execute("VACUUM FULL chunks")
print("  Done")

print("\nChecking database size...")
cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
print(f"  Database size: {cur.fetchone()[0]}")

cur.close()
conn.close()
