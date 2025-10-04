"""SQLModel persistence layer that mirrors the Pydantic schema definitions.

Rather than redefining every attribute, the SQLModel classes inherit from the
Pydantic models defined in ``schemas.schema``. This keeps the database schema
aligned with the structures the rest of the application already consumes.
"""

from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel, Session, select, create_engine

from .schema import (
    Artifact as ArtifactSchema,
    Argument as ArgumentSchema,
    GraphData,
    Quote as QuoteSchema,
    Statement as StatementSchema,
)


class StatementArtifact(SQLModel, table=True):
    __tablename__ = "statement_artifact"

    statement_id: int = Field(foreign_key="statement.id", primary_key=True)
    artifact_id: int = Field(foreign_key="artifact.id", primary_key=True)


class ArgumentPremise(SQLModel, table=True):
    __tablename__ = "argument_premise"

    argument_id: int = Field(foreign_key="argument.id", primary_key=True)
    statement_id: int = Field(foreign_key="statement.id", primary_key=True)


class QuoteModel(SQLModel, QuoteSchema, table=True):
    __tablename__ = "quote"
    model_config = {"from_attributes": True}

    id: Optional[int] = Field(default=None, primary_key=True)

    statement_id: int = Field(foreign_key="statement.id", index=True)
    statement: "StatementModel" = Relationship(back_populates="citations")


class ArtifactModel(SQLModel, ArtifactSchema, table=True):
    __tablename__ = "artifact"
    model_config = {"from_attributes": True}

    id: int = Field(primary_key=True)

    statements: List["StatementModel"] = Relationship(
        back_populates="artifact",
        link_model=StatementArtifact,
    )


class StatementModel(SQLModel, StatementSchema, table=True):
    __tablename__ = "statement"
    model_config = {"from_attributes": True}

    id: int = Field(primary_key=True)

    artifact: List[ArtifactModel] = Relationship(
        back_populates="statements",
        link_model=StatementArtifact,
    )

    citations: List[QuoteModel] = Relationship(
        back_populates="statement",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    arguments_as_premise: List["ArgumentModel"] = Relationship(
        back_populates="premise",
        link_model=ArgumentPremise,
    )

    conclusions_for: List["ArgumentModel"] = Relationship(
        back_populates="conclusion",
    )


class ArgumentModel(SQLModel, ArgumentSchema, table=True):
    __tablename__ = "argument"
    model_config = {"from_attributes": True}

    id: int = Field(primary_key=True)

    premise: List[StatementModel] = Relationship(
        back_populates="arguments_as_premise",
        link_model=ArgumentPremise,
    )

    conclusion_id: int = Field(foreign_key="statement.id", index=True)
    conclusion: StatementModel = Relationship(back_populates="conclusions_for")


def get_session(url: str = "sqlite:///database.db") -> Session:
    """Return a new SQLModel session bound to the configured database URL."""

    engine = create_engine(url, echo=False)
    SQLModel.metadata.create_all(engine)
    return Session(engine)


__all__ = [
    "ArtifactModel",
    "ArgumentModel",
    "QuoteModel",
    "StatementModel",
    "StatementArtifact",
    "ArgumentPremise",
    "GraphData",
    "get_session",
]
