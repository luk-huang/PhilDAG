from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship, Session, select, create_engine

class StatementArtifact(SQLModel, table=True):
    __tablename__ = "statement_artifact"
    statement_id: int = Field(foreign_key="statement.id", primary_key=True)
    artifact_id: int = Field(foreign_key="artifact.id", primary_key=True)


class ArgumentPremise(SQLModel, table=True):
    __tablename__ = "argument_premise"
    argument_id: int = Field(foreign_key="argument.id", primary_key=True)
    statement_id: int = Field(foreign_key="statement.id", primary_key=True)


class Quote(SQLModel, table=True):
    __tablename__ = "quote"
    model_config = {"from_attributes": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    page: int
    text: str

    # A Quote is attached to a Statement (citations for that statement)
    statement_id: int = Field(foreign_key="statement.id", index=True)
    statement: "Statement" = Relationship(back_populates="citations")


class Artifact(SQLModel, table=True):
    __tablename__ = "artifact"
    model_config = {"from_attributes": True}

    id: int = Field(primary_key=True)
    name: str
    author: str
    tile: str   # kept as in your Pydantic model (if this was meant to be "title", rename in both)
    year: str

    # Many-to-many with Statement
    statements: List["Statement"] = Relationship(
        back_populates="artifact",
        link_model=StatementArtifact,
    )


class Statement(SQLModel, table=True):
    __tablename__ = "statement"
    model_config = {"from_attributes": True}

    id: int = Field(primary_key=True)
    statement: str

    # Many-to-many with Artifact (name kept as 'artifact' to mirror your Pydantic model)
    artifact: List[Artifact] = Relationship(
        back_populates="statements",
        link_model=StatementArtifact,
    )

    # One-to-many: a statement has many citations (Quotes)
    citations: List[Quote] = Relationship(
        back_populates="statement",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    # Back-refs for Argument relations
    arguments_as_premise: List["Argument"] = Relationship(
        back_populates="premise",
        link_model=ArgumentPremise,
    )
    conclusions_for: List["Argument"] = Relationship(
        back_populates="conclusion",
    )


class Argument(SQLModel, table=True):
    __tablename__ = "argument"
    model_config = {"from_attributes": True}

    id: int = Field(primary_key=True)
    desc: str

    # Many-to-many premises: Argument.premise <-> Statement.arguments_as_premise
    premise: List[Statement] = Relationship(
        back_populates="arguments_as_premise",
        link_model=ArgumentPremise,
    )

    # One conclusion per argument (many arguments may share the same conclusion statement if desired)
    conclusion_id: int = Field(foreign_key="statement.id", index=True)
    conclusion: Statement = Relationship(back_populates="conclusions_for")


def get_session(url='sqlite://database.db') -> Session:
    """Return a new SQLModel session bound to the global engine."""
    # i haven't tested this
    return Session(create_engine(url, echo=False))


if __name__ == "__main__":
    engine = create_engine("sqlite:///example.db", echo=False)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        if not s.exec(select(Artifact)).first():
            a1 = Artifact(id=1, name="Republic", author="Plato", tile="rep-01", year="~375 BC")
            a2 = Artifact(id=2, name="Nicomachean Ethics", author="Aristotle", tile="eth-01", year="~340 BC")

            st1 = Statement(id=1, statement="Justice is each part doing its own work.")
            st2 = Statement(id=2, statement="Virtue is a habit concerned with choice.")
            st3 = Statement(id=3, statement="The highest good is eudaimonia.")

            # Many-to-many links
            st1.artifact.append(a1)
            st2.artifact.append(a2)
            st3.artifact.append(a2)

            # Citations
            q1 = Quote(page=123, text="Book IV, discussion of justice", statement=st1)
            q2 = Quote(page=15, text="Book II, virtue defined", statement=st2)

            # Argument: premises {st1, st2} ⇒ conclusion st3
            arg = Argument(id=10, desc="From role-based justice and virtue to the highest good",
                        conclusion=st3)
            arg.premise.extend([st1, st2])

            s.add_all([a1, a2, st1, st2, st3, q1, q2, arg])
            s.commit()

    with Session(engine) as s:
        results = (
            s.exec(
                select(Statement)
                .join(Statement.artifact)
                .where(Artifact.author == "Aristotle")
            )
            .all()
        )
        print(f"Results: {results}")
