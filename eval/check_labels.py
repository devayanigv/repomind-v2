import json
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()

qs = json.load(open("eval/queries.json"))
labels = {r.strip().lower() for q in qs for r in q.get("relevant", [])}

cur.execute(
    "SELECT LOWER(full_name) FROM repos WHERE LOWER(full_name) = ANY(%s)",
    (list(labels),)
)
found = {r[0] for r in cur.fetchall()}

missing = sorted(labels - found)

print(f"{len(found)}/{len(labels)} labels in corpus\n")
print("First 10 MISSING repos:")
for m in missing[:10]:
    print(f" - {m}")

cur.close()
conn.close()