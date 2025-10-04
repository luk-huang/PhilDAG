# from PyPDF2 import PdfReader
# MAX_PAGES = 10

# reader = PdfReader("text/Leviathan.pdf")
# counter = 0
# with open("test/Leviathan.txt", "w", encoding="utf-8") as f:
#     for page in reader.pages:
#         f.write(page.extract_text() + "\n")
#         counter += 1
#         if counter >= MAX_PAGES:
#             break
from google import genai
from google.genai import types
import httpx
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional, Dict
from pydantic import BaseModel

class Quote(BaseModel):
    page: int
    text: str

class Premise(BaseModel):
    id: Optional[str] = None
    desc: str
    quotes: List[Quote]
    citations: List[str] = []
    weight: Optional[float] = None
    type: Optional[str] = None               
    derived_from: Optional[List[str]] = None 
    tags: Optional[List[str]] = None


class Argument(BaseModel):
    id: Optional[str] = None
    desc: str                                # summary of reasoning step
    premises: List[str]                      # premise IDs (inputs)
    conclusion: str                          # claim ID (output)
    method: Optional[str] = None             # logical | rhetorical | emotional | analogical | empirical
    form: Optional[str] = None               # deductive | inductive | abductive | analogical
    validity: Optional[float] = None         # 0–1 confidence in reasoning soundness
    quotes: Optional[List[Quote]] = None
    citations: Optional[List[str]] = None
    counterarguments: Optional[List[str]] = None


class Claim(BaseModel):
    id: Optional[str] = None
    desc: str
    weight: Optional[float] = None
    derived_from: Optional[List[str]] = None  # argument IDs supporting this claim
    quotes: Optional[List[Quote]] = None
    citations: Optional[List[str]] = None
    countered_by: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class MetaContext(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    philosophical_tradition: Optional[str] = None  # e.g. existentialism, utilitarianism, rationalism
    summary: Optional[str] = None
    keywords: Optional[List[str]] = None
    notes: Optional[str] = None

class TextAnalysis(BaseModel):
    premises: List[Premise]
    arguments: List[Argument]
    claims: List[Claim]
    metacontext: MetaContext


def extract_arguments(file_path: Path, client: genai.Client):
    with open("services/prompts/extract_arguments.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    uploaded_file = client.files.upload(
        file=file_path,
    )

    output = client.models.generate_content(
        model = "gemini-2.5-flash", 
        contents=[
            uploaded_file,
            prompt
        ],
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=-1
            ),
            response_schema=TextAnalysis
        )
    )

    test_analysis: TextAnalysis = output.parsed
    print(test_analysis)

if __name__ == '__main__':
    client = genai.Client()
    file_path = Path("text/Leviathan.pdf")
    extract_arguments(file_path, client)
