"""Precompute and save contextual passage matrix for fast retrieval."""

import json
import os
import sys
import time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from embedding.encoder import EmbeddingEncoder
from embedding.mapper import DocumentMapper
CHUNKS_PATH = os.path.join(ROOT, "Data", "chunks", "chunks.jsonl")
CACHE_DIR = os.path.join(ROOT, "evaluation", "cache")
MATRIX_PATH = os.path.join(CACHE_DIR, "contextual_passage_matrix.npy")

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(MATRIX_PATH):
        print(f"Matrix already exists at {MATRIX_PATH}")
        mat = np.load(MATRIX_PATH)
        print(f"Loaded matrix shape: {mat.shape}, dtype: {mat.dtype}")
        return

    print("Loading chunks...")
    chunks = [json.loads(line) for line in open(CHUNKS_PATH, encoding="utf-8")]
    print(f"Loaded {len(chunks)} chunks.")
    
    passages = [DocumentMapper.passage_text(c, contextual=True) for c in chunks]
    print("Initializing EmbeddingEncoder...")
    t0 = time.time()
    encoder = EmbeddingEncoder("BAAI/bge-small-en-v1.5", batch_size=64)
    print("Encoding passages...")
    embs = encoder.encode_passages(passages, show_progress=True)
    matrix = np.asarray(embs, dtype=np.float32)
    np.save(MATRIX_PATH, matrix)
    print(f"Successfully saved {matrix.shape} matrix to {MATRIX_PATH} in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
