import os
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
register_vector(conn)
cur = conn.cursor()

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "python library to limit API request rates"
query_vec = model.encode(query, normalize_embeddings=True).tolist()

search_sql = """
    SELECT full_name, description, stars, (embedding <=> %s::vector) AS distance
    FROM repos
    ORDER BY distance ASC
    LIMIT 5;
"""

cur.execute(search_sql, (query_vec,))
results = cur.fetchall()

print(f"\n--- Top 5 Results for: '{query}' ---\n")
for full_name, desc, stars, dist in results:
    print(f"[{round(dist, 4)}] {full_name} ({stars} ⭐)")
    print(f"    {desc}\n")

cur.close()
conn.close()