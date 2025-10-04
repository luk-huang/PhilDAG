from pydantic import BaseModel
from 
class Quote(BaseModel):
    page: int
    text: str

class Artifact(BaseModel):
    id: int
    name: str
    author: str
    title: str
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

class GraphData(BaseModel):
    statements: list[Statement]
    arguments: list[Argument]