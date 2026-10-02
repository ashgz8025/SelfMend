"""
Incident memory store using ChromaDB + sentence-transformers.

Stores past incidents (failure description, root cause, fix applied, outcome)
in a persistent Chroma collection. Retriever returns top-k similar incidents
with similarity scores.

Public API:
    IncidentStore            : main class
    IncidentStore.add(record) -> str
    IncidentStore.retrieve(query, top_k) -> list[IncidentRecord]
    IncidentStore.seed_incidents(incidents) -> None
    get_store() -> IncidentStore      # singleton helper
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ROOT            = Path(__file__).resolve().parents[1]
CHROMA_DIR      = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "incidents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K   = 3


@dataclass
class IncidentRecord:
    """A single incident stored in memory."""
    incident_id: str
    failure_description: str     # natural language description of what failed
    root_cause: str              # why it failed
    fault_type: str              # e.g. "null_spike", "sentinel_reappear"
    fault_class: str             # "schema" | "data_quality" | "distribution"
    affected_column: str
    fix_applied: str             # whitelisted action or "escalated"
    outcome: str                 # "fixed" | "escalated" | "failed"
    similarity_score: float = 0.0   # filled on retrieval
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_document_text(self) -> str:
        """Convert to a single text string for embedding."""
        return (
            f"Failure: {self.failure_description}. "
            f"Root cause: {self.root_cause}. "
            f"Column: {self.affected_column}. "
            f"Fix: {self.fix_applied}. "
            f"Outcome: {self.outcome}."
        )


class IncidentStore:
    """
    Persistent incident memory using ChromaDB + sentence-transformers.

    Uses SentenceTransformerEmbeddingFunction from chromadb.utils to embed
    incident text with all-MiniLM-L6-v2.
    """

    def __init__(
        self,
        persist_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        import chromadb  # noqa: PLC0415
        from chromadb.utils.embedding_functions import (  # noqa: PLC0415
            SentenceTransformerEmbeddingFunction,
        )

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "IncidentStore ready: collection='%s'  n_incidents=%d",
            collection_name, self._collection.count()
        )

    # ── Add ──────────────────────────────────────────────────────────────────

    def add(self, record: IncidentRecord) -> str:
        """
        Add an incident record to the store.
        Returns the incident_id.
        """
        if not record.incident_id:
            record.incident_id = str(uuid.uuid4())

        doc_text = record.to_document_text()
        meta = {
            "fault_type": record.fault_type,
            "fault_class": record.fault_class,
            "affected_column": record.affected_column,
            "fix_applied": record.fix_applied,
            "outcome": record.outcome,
            "root_cause": record.root_cause[:500],     # Chroma meta limit
        }
        meta.update(record.metadata)

        self._collection.upsert(
            ids=[record.incident_id],
            documents=[doc_text],
            metadatas=[meta],
        )
        log.debug("Added incident: %s", record.incident_id)
        return record.incident_id

    # ── Retrieve ─────────────────────────────────────────────────────────────

    def retrieve(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> list[IncidentRecord]:
        """
        Retrieve top-k most similar incidents to `query`.

        Returns list of IncidentRecord with similarity_score set
        (Chroma cosine distance → score = 1 - distance).
        """
        n = min(top_k, self._collection.count())
        if n == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        records: list[IncidentRecord] = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta     = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score    = max(0.0, 1.0 - distance)    # cosine: distance in [0,2], usually [0,1]

            record = IncidentRecord(
                incident_id=doc_id,
                failure_description=results["documents"][0][i],
                root_cause=meta.get("root_cause", ""),
                fault_type=meta.get("fault_type", ""),
                fault_class=meta.get("fault_class", ""),
                affected_column=meta.get("affected_column", ""),
                fix_applied=meta.get("fix_applied", ""),
                outcome=meta.get("outcome", ""),
                similarity_score=round(score, 4),
            )
            records.append(record)

        return records

    # ── Seed ─────────────────────────────────────────────────────────────────

    def seed_incidents(self, incidents: list[dict]) -> None:
        """
        Seed the store with a list of incident dicts.
        Skips incidents whose ID already exists.
        """
        existing_ids = set(self._collection.get()["ids"])
        added = 0
        for inc in incidents:
            iid = inc.get("incident_id", str(uuid.uuid4()))
            if iid in existing_ids:
                continue
            record = IncidentRecord(
                incident_id=iid,
                failure_description=inc["failure_description"],
                root_cause=inc["root_cause"],
                fault_type=inc["fault_type"],
                fault_class=inc["fault_class"],
                affected_column=inc["affected_column"],
                fix_applied=inc["fix_applied"],
                outcome=inc["outcome"],
                metadata=inc.get("metadata", {}),
            )
            self.add(record)
            added += 1
        log.info("Seeded %d new incidents (total: %d)", added, self._collection.count())

    def count(self) -> int:
        return self._collection.count()


# ── Singleton helper ──────────────────────────────────────────────────────────
_store_instance: IncidentStore | None = None


def get_store(reset: bool = False) -> IncidentStore:
    """Return (or create) the shared IncidentStore singleton."""
    global _store_instance
    if _store_instance is None or reset:
        _store_instance = IncidentStore()
    return _store_instance


# ── Seed data ─────────────────────────────────────────────────────────────────

SEED_INCIDENTS: list[dict] = [
    # ── Schema faults ──────────────────────────────────────────────────────
    {
        "incident_id": "inc-001",
        "failure_description": "Column 'Price Per Unit' is missing from the batch. Schema check failed with ExpectColumnToExist.",
        "root_cause": "Upstream ETL job dropped the Price Per Unit column during a schema migration.",
        "fault_type": "drop_column",
        "fault_class": "schema",
        "affected_column": "Price Per Unit",
        "fix_applied": "escalated",
        "outcome": "escalated",
    },
    {
        "incident_id": "inc-002",
        "failure_description": "Column 'Total Spent' was renamed to 'Total_Spent_usd' causing schema validation failure.",
        "root_cause": "A downstream data producer changed the column naming convention without a migration plan.",
        "fault_type": "rename_column",
        "fault_class": "schema",
        "affected_column": "Total Spent",
        "fix_applied": "escalated",
        "outcome": "escalated",
    },
    {
        "incident_id": "inc-003",
        "failure_description": "Quantity column dtype changed from float64 to object/string. Range checks and arithmetic operations fail.",
        "root_cause": "Source system exported Quantity as text instead of numeric due to a serialization bug.",
        "fault_type": "dtype_change",
        "fault_class": "schema",
        "affected_column": "Quantity",
        "fix_applied": "escalated",
        "outcome": "escalated",
    },
    # ── Data quality: null spikes ───────────────────────────────────────────
    {
        "incident_id": "inc-004",
        "failure_description": "Price Per Unit null rate spiked to 50% in this batch. Baseline is approximately 10%. GX ExpectColumnValuesToNotBeNull failed.",
        "root_cause": "POS system experienced a network timeout, resulting in missing price lookups for approximately half of transactions.",
        "fault_type": "null_spike_price",
        "fault_class": "data_quality",
        "affected_column": "Price Per Unit",
        "fix_applied": "impute_nulls",
        "outcome": "fixed",
        "metadata": {"baseline_null_rate": 0.10, "observed_null_rate": 0.50},
    },
    {
        "incident_id": "inc-005",
        "failure_description": "Quantity null rate rose to 50%. GX not-null check failed. Baseline is approximately 8%.",
        "root_cause": "Barcode scanner failure caused transaction records to be saved without item quantities.",
        "fault_type": "null_spike_quantity",
        "fault_class": "data_quality",
        "affected_column": "Quantity",
        "fix_applied": "impute_nulls",
        "outcome": "fixed",
        "metadata": {"baseline_null_rate": 0.08, "observed_null_rate": 0.50},
    },
    # ── Data quality: sentinels ─────────────────────────────────────────────
    {
        "incident_id": "inc-006",
        "failure_description": "Sentinel strings 'ERROR' and 'UNKNOWN' re-appeared in the Item column at approximately 20% of rows. GX category check failed.",
        "root_cause": "ETL preprocessing step was skipped in this batch run, allowing raw sentinel values to pass through.",
        "fault_type": "sentinel_reappear",
        "fault_class": "data_quality",
        "affected_column": "Item",
        "fix_applied": "normalize_sentinels_to_null",
        "outcome": "fixed",
        "metadata": {"sentinel_rate": 0.20},
    },
    {
        "incident_id": "inc-007",
        "failure_description": "Payment Method column contains 'ERROR' sentinel in 20% of rows. Expected values are Cash, Credit Card, Digital Wallet.",
        "root_cause": "Payment gateway integration returned error codes that were not filtered before ingestion.",
        "fault_type": "sentinel_payment",
        "fault_class": "data_quality",
        "affected_column": "Payment Method",
        "fix_applied": "normalize_sentinels_to_null",
        "outcome": "fixed",
        "metadata": {"sentinel_rate": 0.20},
    },
    # ── Data quality: Total mismatch ────────────────────────────────────────
    {
        "incident_id": "inc-008",
        "failure_description": "Total Spent does not match Quantity x Price Per Unit for approximately 30% of rows. Mismatch exceeds 1% tolerance.",
        "root_cause": "A discount code was applied to Total Spent without updating the unit price, causing a systematic derivation mismatch.",
        "fault_type": "total_mismatch",
        "fault_class": "data_quality",
        "affected_column": "Total Spent",
        "fix_applied": "restore_total_from_quantity_price",
        "outcome": "fixed",
        "metadata": {"mismatch_rate": 0.30},
    },
    {
        "incident_id": "inc-009",
        "failure_description": "Total Spent values are inconsistent with derived Quantity times Price. Custom integrity check reports 25% mismatch rate.",
        "root_cause": "Concurrent writes from two POS systems caused race condition resulting in partially overwritten Total values.",
        "fault_type": "total_mismatch",
        "fault_class": "data_quality",
        "affected_column": "Total Spent",
        "fix_applied": "restore_total_from_quantity_price",
        "outcome": "fixed",
        "metadata": {"mismatch_rate": 0.25},
    },
    # ── Data quality: out-of-range ──────────────────────────────────────────
    {
        "incident_id": "inc-010",
        "failure_description": "Quantity values range from 50 to 200 in this batch. Expected range is 1 to 5. GX range check failed.",
        "root_cause": "A bulk catering order was incorrectly mixed into the retail transaction batch.",
        "fault_type": "out_of_range_quantity",
        "fault_class": "data_quality",
        "affected_column": "Quantity",
        "fix_applied": "clip_outliers",
        "outcome": "fixed",
        "metadata": {"observed_max": 200, "expected_max": 5},
    },
    # ── Data quality: duplicates ────────────────────────────────────────────
    {
        "incident_id": "inc-011",
        "failure_description": "20% duplicate rows detected in this batch. Transaction IDs are repeated.",
        "root_cause": "The message queue consumer processed events twice due to an at-least-once delivery configuration.",
        "fault_type": "duplicate_rows",
        "fault_class": "data_quality",
        "affected_column": "Transaction ID",
        "fix_applied": "drop_duplicates",
        "outcome": "fixed",
        "metadata": {"duplicate_rate": 0.20},
    },
    # ── Distribution faults ─────────────────────────────────────────────────
    {
        "incident_id": "inc-012",
        "failure_description": "Evidently drift detected: Quantity mean shifted from approximately 3.0 to approximately 5.5. GX range check passed. KS-test p-value near zero.",
        "root_cause": "Airport location opened a new large-group seating area, causing bulk orders to inflate average quantity.",
        "fault_type": "mean_shift_quantity",
        "fault_class": "distribution",
        "affected_column": "Quantity",
        "fix_applied": "escalated",
        "outcome": "escalated",
        "metadata": {"original_mean": 3.0, "shifted_mean": 5.5},
    },
    {
        "incident_id": "inc-013",
        "failure_description": "Price Per Unit distribution shifted to only high-price items (5.0, 6.0, 7.5). Low-price items disappeared. Evidently chi-square drift detected.",
        "root_cause": "Menu update removed budget items; only premium items remain, shifting the price distribution upward.",
        "fault_type": "mean_shift_price",
        "fault_class": "distribution",
        "affected_column": "Price Per Unit",
        "fix_applied": "escalated",
        "outcome": "escalated",
        "metadata": {"price_values": [5.0, 6.0, 7.5]},
    },
    {
        "incident_id": "inc-014",
        "failure_description": "Item distribution is severely skewed: 100% of items are 'Coffee'. Evidently chi-square detected. GX InSet check passed since Coffee is a valid category.",
        "root_cause": "A data export filter accidentally selected only coffee transactions from the source system.",
        "fault_type": "skewed_item_mix",
        "fault_class": "distribution",
        "affected_column": "Item",
        "fix_applied": "escalated",
        "outcome": "escalated",
        "metadata": {"dominant_item": "Coffee", "share": 1.0},
    },
    # ── Additional edge cases ───────────────────────────────────────────────
    {
        "incident_id": "inc-015",
        "failure_description": "Multiple columns drifted simultaneously: Quantity mean shifted and Item mix became Coffee-dominant. Both Evidently KS and chi-square tests flagged.",
        "root_cause": "A single location's data feed went offline; only automated coffee machine transactions (quantity=1) were recorded.",
        "fault_type": "skewed_item_mix",
        "fault_class": "distribution",
        "affected_column": "Item",
        "fix_applied": "escalated",
        "outcome": "escalated",
        "metadata": {"n_columns_drifted": 2},
    },
    {
        "incident_id": "inc-016",
        "failure_description": "Null spike in both Price Per Unit and Quantity simultaneously. GX not-null checks failed on both columns.",
        "root_cause": "Database connection pool exhaustion caused incomplete transaction records for a 2-hour window.",
        "fault_type": "null_spike_price",
        "fault_class": "data_quality",
        "affected_column": "Price Per Unit",
        "fix_applied": "impute_nulls",
        "outcome": "fixed",
        "metadata": {"columns_affected": ["Price Per Unit", "Quantity"]},
    },
    {
        "incident_id": "inc-017",
        "failure_description": "Location column contains 'ERROR' sentinel values. GX allowed-categories check failed.",
        "root_cause": "New store location code was not registered in the lookup table before transactions were recorded.",
        "fault_type": "sentinel_reappear",
        "fault_class": "data_quality",
        "affected_column": "Location",
        "fix_applied": "normalize_sentinels_to_null",
        "outcome": "fixed",
        "metadata": {"sentinel": "ERROR", "column": "Location"},
    },
    {
        "incident_id": "inc-018",
        "failure_description": "Quantity values are stored as strings ('1', '2', '3') instead of numeric. GX range check cannot evaluate. Custom dtype check failed.",
        "root_cause": "CSV export from legacy POS system omitted numeric type annotations, causing downstream CSV parser to infer string dtype.",
        "fault_type": "dtype_change",
        "fault_class": "schema",
        "affected_column": "Quantity",
        "fix_applied": "escalated",
        "outcome": "escalated",
        "metadata": {"observed_dtype": "object", "expected_dtype": "float64"},
    },
    {
        "incident_id": "inc-019",
        "failure_description": "Total Spent is null for approximately 40% of rows but Quantity and Price Per Unit are available. Can recompute.",
        "root_cause": "ETL total-computation step failed silently for one location's data partition.",
        "fault_type": "total_mismatch",
        "fault_class": "data_quality",
        "affected_column": "Total Spent",
        "fix_applied": "restore_total_from_quantity_price",
        "outcome": "fixed",
        "metadata": {"null_rate_total": 0.40, "derivable": True},
    },
    {
        "incident_id": "inc-020",
        "failure_description": "Duplicate transaction IDs detected at 15% rate. Downstream aggregation double-counts revenue.",
        "root_cause": "Retry logic in the transaction API created duplicate records without idempotency keys.",
        "fault_type": "duplicate_rows",
        "fault_class": "data_quality",
        "affected_column": "Transaction ID",
        "fix_applied": "drop_duplicates",
        "outcome": "fixed",
        "metadata": {"duplicate_rate": 0.15},
    },
]
