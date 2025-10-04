from fastapi import APIRouter, UploadFile
from services.process import analyze
from pathlib import Path
import os

router = APIRouter()

@router.post("/")
async def analyze_work(file: UploadFile):
    path = Path(f"uploaded_files/{file.filename}")
    os.makedirs("uploaded_files", exist_ok=True)
    with open(path, "wb") as f:
        f.write(await file.read())

    result = analyze(path)

    return result.model_dump()