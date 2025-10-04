from pydantic import BaseModel

class Quote(BaseModel):
    page: int
    text: str

class Artifact(BaseModel):
    id: int
    name: str
    author: str
    tile: str
    year: str

class Statement(BaseModel):
    id: int
    artifact: list[Artifact]
    statement: str
    citations: list[Quote] = []

class Argument(BaseModel):
    id: int
    premise: list[Statement]
    conclusion: Statement
    desc: str