"""
Evaluation Runner for RepoMind v2.
Calculates Recall@5 and MRR for Baseline Keyword Search vs Vector Search.
"""
import json
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

# String normalization helper
norm = lambda s: s.strip().lower().removeprefix("https://github.com/")

# Load evaluation queries
with open("eval/queries.json", "r") as f:
    queries = json.load(f)

def run_baseline_search(query_str, top_k=5):
    """Simple SQL LIKE / Keyword search baseline."""
    sql = """
        SELECT full_name FROM repos
        WHERE full_name ILIKE %s OR description ILIKE %s OR readme ILIKE %s
        LIMIT %s;
    """
    pattern = f"%{query_str}%"
    cur.execute(sql, (pattern, pattern, pattern, top_k))
    return [r[0] for r in cur.fetchall()]

def run_vector_search(query_str, top_k=5):
    """Vector similarity search using HNSW cosine distance."""
    query_vec = model.encode(query_str, normalize_embeddings=True).tolist()
    sql = """
        SELECT full_name FROM repos
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s;
    """
    cur.execute(sql, (query_vec, top_k))
    return [r[0] for r in cur.fetchall()]

def evaluate(search_fn):
    total_recall = 0.0
    total_mrr = 0.0
    valid_queries = 0

    for q in queries:
        relevant_set = set(norm(r) for r in q.get("relevant", []))
        if not relevant_set:
            continue

        valid_queries += 1
        retrieved = [norm(r) for r in search_fn(q["query"], top_k=5)]

        # Calculate Recall@5
        hits = sum(1 for r in retrieved if r in relevant_set)
        recall = hits / len(relevant_set)
        total_recall += recall

        # Calculate MRR
        mrr = 0.0
        for rank, r in enumerate(retrieved, start=1):
            if r in relevant_set:
                mrr = 1.0 / rank
                break
        total_mrr += mrr

    if valid_queries == 0:
        return 0.0, 0.0

    return total_recall / valid_queries, total_mrr / valid_queries

if __name__ == "__main__":
    print("--- Running Evaluation Benchmark ---")
    b_recall, b_mrr = evaluate(run_baseline_search)
    v_recall, v_mrr = evaluate(run_vector_search)

    print(f"\nBaseline Search -> Recall@5: {b_recall:.4f} | MRR: {b_mrr:.4f}")
    print(f"Vector Search   -> Recall@5: {v_recall:.4f} | MRR: {v_mrr:.4f}")

    cur.close()
    conn.close()