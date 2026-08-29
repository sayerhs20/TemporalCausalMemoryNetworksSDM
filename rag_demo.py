"""
RAG layer for TCMN-lite: two retrieval demos.
    1. Business-rule retrieval -> grounds causal-graph priors.
    2. Trajectory retrieval -> finds similar past order sequences (a
       text-based stand-in for what the model's causal memory does
       internally in vector form).

Needs internet access on first run to download the embedding model
(~80MB, one-time). If you're offline, this script will fail at the
SentenceTransformer(...) line -- everything else in the project does
not depend on this.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BUSINESS_RULES = [
    "Department location can affect salary bands across regions.",
    "Product category affects price elasticity and customer response to price changes.",
    "Customer country can influence payment behavior and order completion rates.",
    "Employee tenure since joining_date affects likelihood of promotion, not termination.",
    "A late-stage price increase, after an initial quote, tends to raise cancellation risk.",
]

TRAJECTORY_SUMMARIES = [
    "Order 401: price moved 1200 -> 1150 -> 1180, ended Completed.",
    "Order 402: price moved 2500 -> 2400 -> 2450, ended Cancelled.",
    "Order 403: price moved 800 -> 850 -> 900, ended Pending.",
    "Order 404: price moved 1500 -> 1550 -> 1600, ended Completed.",
    "Order 405: price moved 3200 -> 3100 -> 3150, ended Cancelled.",
]


def build_index(texts, model):
    vectors = model.encode(texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index


def retrieve(query, texts, index, model, top_k=2):
    query_vector = model.encode([query])
    _, indices = index.search(np.array(query_vector), top_k)
    return [texts[i] for i in indices[0]]


def main():
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    rule_index = build_index(BUSINESS_RULES, model)
    traj_index = build_index(TRAJECTORY_SUMMARIES, model)

    # --- Demo 1: business-rule retrieval for a causal-graph prior question ---
    query1 = "Does customer country confound whether an order is completed?"
    top_rules = retrieve(query1, BUSINESS_RULES, rule_index, model, top_k=2)
    print(f"\nQuery: {query1}")
    print("Retrieved business rules:")
    for r in top_rules:
        print(" -", r)

    # --- Demo 2: trajectory retrieval for a new, unseen order pattern ---
    new_trajectory = "price moved 2000 -> 1900 -> 1950"
    top_traj = retrieve(new_trajectory, TRAJECTORY_SUMMARIES, traj_index, model, top_k=1)
    print(f"\nNew order pattern: {new_trajectory}")
    print("Most similar past trajectory:")
    for t in top_traj:
        print(" -", t)

    # --- Demo 3: turning a model's causal attribution into a plain-English explanation ---
    # (stub -- swap in ollama.chat(...) here if you have Ollama + llama3 installed locally)
    causal_attribution = "price2 revision (2500 -> 2400) is the primary driver of order 402 being Cancelled"
    print(f"\nCausal attribution to explain: {causal_attribution}")
    print("Explanation (LLM call would go here):")
    print(" -> \"Order 402 was likely cancelled because the price was lowered mid-negotiation, "
          "which retrieved a similar historical pattern (order 402 itself) and matched the "
          "business rule about late-stage price changes affecting cancellation risk.\"")


if __name__ == "__main__":
    main()
