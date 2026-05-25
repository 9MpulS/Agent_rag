"""SQLAlchemy declarative models for the RAG agent database."""

import enum

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    event,
    DDL,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

from agent_rag.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class DocType(enum.Enum):
    """Types of university documents."""
    regulation = "regulation"
    order = "order"
    instruction = "instruction"
    manual = "manual"


class RegistrySection(Base):
    """Registry section representing a category of documents."""

    __tablename__ = "registry_sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    description_embedding = mapped_column(
        Vector(settings.EMBEDDING_DIM), nullable=True
    )

    # Relationships
    documents: Mapped[list["Document"]] = relationship(
        back_populates="registry_section", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="registry_section"
    )

    def __repr__(self) -> str:
        return f"<RegistrySection(id={self.id}, name='{self.name}')>"


class Document(Base):
    """A university document belonging to a registry section."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    registry_section_id: Mapped[int] = mapped_column(
        ForeignKey("registry_sections.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    doc_type: Mapped[DocType] = mapped_column(
        Enum(DocType, name="doc_type_enum"), nullable=False
    )
    url: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Relationships
    registry_section: Mapped["RegistrySection"] = relationship(
        back_populates="documents"
    )
    pages: Mapped[list["Page"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, title='{self.title}')>"


class Page(Base):
    """A single page from a document."""

    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="pages")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Page(id={self.id}, doc_id={self.document_id}, page={self.page_number})>"


class Chunk(Base):
    """A text chunk for retrieval, with embedding and TSVector."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    page_id: Mapped[int] = mapped_column(
        ForeignKey("pages.id"), nullable=False
    )
    registry_section_id: Mapped[int] = mapped_column(
        ForeignKey("registry_sections.id"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_stemmed: Mapped[str] = mapped_column(Text, nullable=True)
    embedding = mapped_column(Vector(settings.EMBEDDING_DIM), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content_tsv = mapped_column(TSVECTOR, nullable=True)

    # Relationships
    page: Mapped["Page"] = relationship(back_populates="chunks")
    registry_section: Mapped["RegistrySection"] = relationship(
        back_populates="chunks"
    )

    __table_args__ = (
        # GIN index for full-text search
        Index("ix_chunk_content_tsv_gin", "content_tsv", postgresql_using="gin"),
        # B-tree index for section filtering
        Index("ix_chunk_section_id", "registry_section_id"),
    )

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, index={self.chunk_index}, len={len(self.content)})>"


# HNSW index for vector similarity search (cosine distance)
# Created as a separate Index because pgvector uses custom ops
hnsw_index = Index(
    "ix_chunk_embedding_hnsw",
    Chunk.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


# SQL trigger for auto-updating content_tsv on INSERT/UPDATE
TSV_TRIGGER_FUNCTION = DDL("""
CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('simple', COALESCE(NEW.content_stemmed, NEW.content));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
""")

TSV_TRIGGER = DDL("""
CREATE TRIGGER chunks_content_tsv_update
    BEFORE INSERT OR UPDATE OF content ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION chunks_tsv_trigger();
""")

# Register DDL events to create trigger after table creation
event.listen(Chunk.__table__, "after_create", TSV_TRIGGER_FUNCTION)
event.listen(Chunk.__table__, "after_create", TSV_TRIGGER)
