from dataclasses import dataclass

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from private_gpt.di import root_injector
from private_gpt.retrieval.retrieval_service import RetrievalService

retrieval_router = APIRouter()


@dataclass
class RetrieveBody(BaseModel):
    query: str
    limit: int = 10
    context_size: int = 0


@retrieval_router.post("/retrieve")
async def retrieve(body: RetrieveBody) -> JSONResponse:
    service = root_injector.get(RetrievalService)
    results = await service.retrieve_relevant_nodes(
        body.query, body.limit, body.context_size
    )
    return JSONResponse(content=jsonable_encoder(results))
