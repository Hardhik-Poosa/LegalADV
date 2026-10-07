"""Initial schema — legal_documents, legal_sections, section_chunks.

Revision: 0001
Creates:
  * pgvector extension
  * legal_documents     (with versioning, jurisdiction, effective_from, repealed_on)
  * legal_sections      (with effective_from, repealed_on, is_repealed flag)
  * section_chunks      (Vector HNSW index + tsvector GIN index + trigger)
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from app.models.legal import EMBEDDING_DIM  # single source of truth for dim

# Alembic revision identifiers.
revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    # ── 0. pgvector extension ─────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── 1. legal_documents ────────────────────────────────────────────────────
    op.create_table(
        "legal_documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("act_name", sa.String(255), nullable=False),
        sa.Column("full_title", sa.String(512), nullable=True),
        sa.Column("act_number", sa.String(50), nullable=True),
        sa.Column("jurisdiction", sa.String(10), nullable=False),
        sa.Column("version", sa.SmallInteger(), nullable=False, server_default="1"),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("repealed_on", sa.Date(), nullable=True),
        sa.Column("source_url", sa.String(2048), nullable=True),
        sa.Column("checksum", sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_legal_documents"),
        sa.UniqueConstraint(
            "act_name", "jurisdiction", "version",
            name="uq_document_act_jur_ver",
        ),
    )
    op.create_index("ix_legal_documents_act_name", "legal_documents", ["act_name"])
    op.create_index("ix_legal_documents_jurisdiction", "legal_documents", ["jurisdiction"])
    op.create_index(
        "ix_legal_documents_effective_from", "legal_documents", ["effective_from"]
    )

    # ── 2. legal_sections ─────────────────────────────────────────────────────
    op.create_table(
        "legal_sections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "document_id",
            sa.Integer(),
            sa.ForeignKey("legal_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("section_number", sa.String(20), nullable=False),
        sa.Column("heading", sa.String(512), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("is_repealed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("repealed_on", sa.Date(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_legal_sections"),
    )
    op.create_index("ix_legal_sections_document_id", "legal_sections", ["document_id"])
    op.create_index(
        "ix_legal_sections_section_number", "legal_sections", ["section_number"]
    )
    op.create_index(
        "ix_legal_sections_effective_from_repealed_on",
        "legal_sections",
        ["effective_from", "repealed_on"],
    )

    # ── 3. section_chunks ─────────────────────────────────────────────────────
    op.create_table(
        "section_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "section_id",
            sa.Integer(),
            sa.ForeignKey("legal_sections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.SmallInteger(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("tsv", sa.dialects.postgresql.TSVECTOR(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_section_chunks"),
    )

    # HNSW index for cosine similarity (pgvector >= 0.5.0)
    op.execute(
        f"""
        CREATE INDEX ix_section_chunks_embedding_hnsw
            ON section_chunks
         USING hnsw (embedding vector_cosine_ops)
          WITH (m = 16, ef_construction = 64)
        """
    )
    # GIN index for tsvector full-text search
    op.execute(
        "CREATE INDEX ix_section_chunks_tsv_gin ON section_chunks USING gin (tsv)"
    )
    op.create_index("ix_section_chunks_section_id", "section_chunks", ["section_id"])

    # ── 4. tsvector auto-update trigger ──────────────────────────────────────
    # Keeps tsv consistent on INSERT/UPDATE without requiring app-level logic,
    # while the loader also sets it explicitly for bulk performance.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION trg_section_chunks_tsv()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            NEW.tsv := to_tsvector('english', COALESCE(NEW.text, ''));
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_section_chunks_tsv_update
        BEFORE INSERT OR UPDATE OF text ON section_chunks
        FOR EACH ROW EXECUTE FUNCTION trg_section_chunks_tsv()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_section_chunks_tsv_update ON section_chunks")
    op.execute("DROP FUNCTION IF EXISTS trg_section_chunks_tsv()")
    op.execute("DROP INDEX IF EXISTS ix_section_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_section_chunks_tsv_gin")
    op.drop_table("section_chunks")
    op.drop_table("legal_sections")
    op.drop_table("legal_documents")
    # Keep the pgvector extension; other tables may use it.
