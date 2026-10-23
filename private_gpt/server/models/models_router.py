from fastapi import APIRouter, Depends, Query, Request

from private_gpt.chat.input_models import ModelInfoOutput, ModelListOutput
from private_gpt.server.models.models_service import ModelsService
from private_gpt.server.utils.auth import authenticated

models_router = APIRouter(
    prefix="/v1",
    dependencies=[Depends(authenticated)],
    tags=["Models"],
    responses={401: {"description": "Unauthorized"}},
)


@models_router.get(
    "/models",
    response_model=ModelListOutput,
    summary="List Models",
    tags=["Models"],
)
async def list_models(
    request: Request,
    before_id: str | None = Query(default=None),
    after_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> ModelListOutput:
    models_service: ModelsService = request.state.injector.get(ModelsService)
    return models_service.list_models(before_id, after_id, limit)


@models_router.get(
    "/models/{model_id}",
    response_model=ModelInfoOutput,
    summary="Get a Model",
    tags=["Models"],
)
async def get_model(
    request: Request,
    model_id: str,
) -> ModelInfoOutput:
    models_service: ModelsService = request.state.injector.get(ModelsService)
    return models_service.get_model(model_id)
