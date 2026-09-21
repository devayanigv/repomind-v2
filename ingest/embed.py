"""
Batch Embedding Generator for RepoMind v2.
Generates MiniLM vector embeddings for repos and updates Postgres using psycopg2 execute_values.
"""
import os
import time
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from .env file!")

BATCH_SIZE = 64  # Fetch & encode in batches of 64


def run_embedding():
    print("--- Loading Sentence Transformer Model ---")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    cur = conn.cursor()

    # Get total count of pending embeddings
    cur.execute("SELECT COUNT(*) FROM repos WHERE embedding IS NULL;")
    total_pending = cur.fetchone()[0]
    print(f"Total repositories to embed: {total_pending}")

    if total_pending == 0:
        print("All repositories already have embeddings!")
        cur.close()
        conn.close()
        return

    processed = 0
    start_time = time.time()

    update_query = """
        UPDATE repos SET embedding = data.emb::vector
        FROM (VALUES %s) AS data(id, emb)
        WHERE repos.id = data.id;
    """

    while True:
        # Fetch batch of rows missing embeddings
        cur.execute(
            "SELECT id, description, readme FROM repos WHERE embedding IS NULL LIMIT %s;",
            (BATCH_SIZE,)
        )
        rows = cur.fetchall()

        if not rows:
            break

        ids = [row[0] for row in rows]
        texts = []

        for row in rows:
            desc = row[1] or ""
            readme = (row[2] or "")[:1200]  # Truncate README to 1200 chars
            combined_text = f"{desc} {readme}".strip()
            # Fallback text if both description and readme are empty
            texts.append(combined_text if combined_text else "repository")

        # Encode with normalization for cosine distance
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        # Convert numpy vectors to list format for pgvector casting
        data_tuples = [(rid, emb.tolist()) for rid, emb in zip(ids, embeddings)]

        # Fast bulk update using execute_values
        execute_values(cur, update_query, data_tuples)
        conn.commit()

        processed += len(rows)
        print(f"[{processed}/{total_pending}] Embeddings saved to database...")

    elapsed = round(time.time() - start_time, 2)
    print(f"\nEmbedding completed in {elapsed} seconds!")

    cur.close()
    conn.close()


if __name__ == "__main__":
    run_embedding()