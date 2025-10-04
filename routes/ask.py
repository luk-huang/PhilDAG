from fastapi import APIRouter

from schemas.schema import AskPhilRequest, AskPhilResponse, GraphData
from services.general_user_queries import query_graph

router = APIRouter()

@router.post("/", response_model=AskPhilResponse)
async def ask_phil(payload: AskPhilRequest):
    # resolve question -> answer + subgraph
    graph = payload.graph
    deepdag = payload.deepdag
    question = payload.question
    highlight_claims, highlight_arguments, answer = query_graph(
        query=question,
        claims=graph.statements,
        arguments=graph.arguments,
        prefilter=not deepdag,
    )
    return AskPhilResponse(
        answer=answer,
        subgraph=GraphData(
            statements=highlight_claims,
            arguments=highlight_arguments
        )
    )
