import json
import os
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_NZw4upP1hyWA@ep-lucky-field-aupwacod-pooler.c-10.us-east-1.aws.neon.tech/neondb?sslmode=require")
conn = psycopg2.connect(db_url)
register_vector(conn)
cur = conn.cursor()

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("eval/queries.json", "r") as f:
    queries = json.load(f)

print(f"--- Semantic Corpus Search ({len(queries)} Queries) ---\n")

for q in queries:
    query_str = q["query"]
    query_vec = model.encode(query_str, normalize_embeddings=True).tolist()
    
    sql = """
        SELECT full_name, description FROM repos
        ORDER BY embedding <=> %s::vector ASC
        LIMIT 3;
    """
    cur.execute(sql, (query_vec,))
    results = cur.fetchall()
    
    print(f"Query #{q['id']} ({q['type']}): \"{query_str}\"")
    for repo, desc in results:
        short_desc = (desc[:65] + "...") if desc else "No description"
        print(f"  -> \"{repo}\" ({short_desc})")
    print("-" * 60)

cur.close()
conn.close()