from fastapi import APIRouter
from schemas.schema import AskPhilResponse, AskPhilRequest
from services.general_user_queries import query_graph

router = APIRouter()

@router.post("/", response_model=AskPhilResponse)
async def ask_phil(payload: AskPhilRequest):
    # resolve question → answer + subgraph
    return AskPhilResponse(answer=..., focus_statements=..., focus_arguments=...