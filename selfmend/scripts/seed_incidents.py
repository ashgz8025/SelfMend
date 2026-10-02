"""Seed the ChromaDB incident store with the 20 predefined incidents."""
import sys, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from memory.incident_store import get_store, SEED_INCIDENTS

store = get_store()
store.seed_incidents(SEED_INCIDENTS)
print(f"\nTotal incidents in store: {store.count()}")

# Quick retrieval test
print("\nRetrieval test — query: 'null spike in price column'")
results = store.retrieve("null spike in price column", top_k=3)
for r in results:
    print(f"  [{r.similarity_score:.3f}] {r.incident_id}: {r.failure_description[:80]}...")
