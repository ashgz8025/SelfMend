"""
SQLAlchemy ORM models for the SelfMend audit log.

Tables:
  pipeline_runs    : one row per pipeline run
  check_results    : GX + drift checks per run
  incidents        : incidents recorded in ChromaDB (mirrored for query)
  diagnoses        : LLM diagnosis + faithfulness per run
  decisions        : auto_fix / escalate decision per run
  remediations     : remediation action result per run
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    run_id      = Column(String(64), unique=True, nullable=False, index=True)
    batch_id    = Column(String(128), nullable=False)
    created_at  = Column(DateTime(timezone=True), default=_utcnow)
    has_failures = Column(Boolean, default=False)
    gx_passed   = Column(Boolean, default=True)
    drift_detected = Column(Boolean, default=False)
    decision    = Column(String(32))    # "auto_fix" | "escalate" | "clean"
    status      = Column(String(32), default="pending")  # pending|running|done|error

    check_results = relationship("CheckResult", back_populates="run", cascade="all, delete-orphan")
    diagnoses     = relationship("Diagnosis",   back_populates="run", cascade="all, delete-orphan")
    decisions_rel = relationship("Decision",    back_populates="run", cascade="all, delete-orphan")
    remediations  = relationship("Remediation", back_populates="run", cascade="all, delete-orphan")


class CheckResult(Base):
    __tablename__ = "check_results"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    run_id      = Column(String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True)
    layer       = Column(String(16))    # "gx" | "drift"
    check_type  = Column(String(128))
    column_name = Column(String(128))
    severity    = Column(String(16))    # "high" | "medium" | "low"
    success     = Column(Boolean)
    detail      = Column(JSON)
    created_at  = Column(DateTime(timezone=True), default=_utcnow)

    run = relationship("PipelineRun", back_populates="check_results")


class Incident(Base):
    __tablename__ = "incidents"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    incident_id         = Column(String(64), unique=True, nullable=False, index=True)
    run_id              = Column(String(64), nullable=True, index=True)
    failure_description = Column(Text)
    root_cause          = Column(Text)
    fault_type          = Column(String(64))
    fault_class         = Column(String(32))
    affected_column     = Column(String(128))
    fix_applied         = Column(String(128))
    outcome             = Column(String(32))
    created_at          = Column(DateTime(timezone=True), default=_utcnow)


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    run_id           = Column(String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True)
    root_cause       = Column(Text)
    proposed_action  = Column(String(128))
    confidence       = Column(Float)
    risk_level       = Column(String(16))
    faithfulness_score = Column(Float)
    faithfulness_passed = Column(Boolean)
    evidence_used    = Column(JSON)     # list of incident IDs cited
    retrieved_incidents = Column(JSON)  # full top-k results
    diagnosis_raw    = Column(Text)
    created_at       = Column(DateTime(timezone=True), default=_utcnow)

    run = relationship("PipelineRun", back_populates="diagnoses")


class Decision(Base):
    __tablename__ = "decisions"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    run_id         = Column(String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True)
    decision       = Column(String(32))    # "auto_fix" | "escalate"
    reason         = Column(Text)
    faithfulness_score = Column(Float)
    created_at     = Column(DateTime(timezone=True), default=_utcnow)

    run = relationship("PipelineRun", back_populates="decisions_rel")


class Remediation(Base):
    __tablename__ = "remediations"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    run_id         = Column(String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True)
    action         = Column(String(128))
    success        = Column(Boolean)
    promoted       = Column(Boolean, default=False)
    pre_check      = Column(JSON)     # detection report before fix
    post_check     = Column(JSON)     # detection report after fix
    reason         = Column(Text)
    created_at     = Column(DateTime(timezone=True), default=_utcnow)

    run = relationship("PipelineRun", back_populates="remediations")
