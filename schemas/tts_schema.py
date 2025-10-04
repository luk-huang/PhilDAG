from pydantic import BaseModel
from typing import Optional


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None  # Defaults to Alice in the route
    model_id: Optional[str] = None  # Defaults to eleven_monolingual_v1
    voice_settings: Optional[dict] = None