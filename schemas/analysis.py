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


class Claim(BaseModel):
    id: int
    artifact: Artifact
    desc: str
    quotes: list[Quote] = []
    citations: list[str] = []

class Argument(BaseModel):
    id: int
    premise: list[Claim]
    conclusion: Claim
    desc: str