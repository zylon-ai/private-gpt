from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from private_gpt.di import root_injector
from private_gpt.retrieval.retrieval_service import RetrievalService

retrieval_router = APIRouter()


@retrieval_router.get("/retrieve")
async def retrieve(query: str, limit: int = 10, context_size: int = 0) -> JSONResponse:
    service = root_injector.get(RetrievalService)
    results = await service.retrieve_relevant_nodes(query, limit, context_size)
    return JSONResponse(content=jsonable_encoder(results))
