import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()
cur.execute("DELETE FROM chunks WHERE book_name NOT IN ('claude_synthetic', 'stackexchange', 'cot_reasoning')")
print(f"Deleted {cur.rowcount} book chunks")
conn.commit()
conn.close()
