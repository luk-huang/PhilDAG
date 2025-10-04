from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import os

from schemas.tts_schema import TTSRequest

router = APIRouter()

# Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Alice voice ID (pre-made voice)
ALICE_VOICE_ID = "NOpBlnGInO9m6vDvFkFC"


@router.post("/")
async def text_to_speech(payload: TTSRequest):
    """
    Convert text to speech using ElevenLabs API with Alice voice
    """
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")
    
    # Use Alice voice by default
    # voice_id = payload.voice_id or ALICE_VOICE_ID
    voice_id = ALICE_VOICE_ID
    
    # Default voice settings for natural speech
    voice_settings = payload.voice_settings or {
        "stability": 0.5,
        "similarity_boost": 0.75,
        "speed": 0.85
    }
    
    # Prepare the request payload
    request_payload = {
        "text": payload.text,
        "model_id": payload.model_id or "eleven_multilingual_v2",
        "voice_settings": voice_settings
    }
    
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ELEVENLABS_API_URL}/{voice_id}",
                json=request_payload,
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"ElevenLabs API error: {response.text}"
                )
            
            # Return audio as streaming response
            return StreamingResponse(
                iter([response.content]),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "attachment; filename=speech.mp3"
                }
            )
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to ElevenLabs timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")