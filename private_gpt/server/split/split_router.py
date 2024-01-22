from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from private_gpt.server.split.split_service import (
    Split,
    SplitService,
)
from private_gpt.server.utils.auth import authenticated

split_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class SplitTextBody(BaseModel):
    text: str = Field(
        examples=[
            "Avatar is set in an Asian and Arctic-inspired world in which some "
            "people can telekinetically manipulate one of the four elements—water, "
            "earth, fire or air—through practices known as 'bending', inspired by "
            "Chinese martial arts."
        ]
    )
    chunk_size: int = Field(
        400,
        description="The maximum number of characters in each split chunk.",
        examples=[400],
    )


class SplitResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[Split]


@split_router.post("/split", tags=["Split"])
def split_generation(request: Request, body: SplitTextBody) -> SplitResponse:
    """Get semantically split chunks of a given text input.

    The split chunks can be easily used for prompt augmentation.
    """
    service = request.state.injector.get(SplitService)
    splits = service.texts_split(body.text, body.chunk_size)
    return SplitResponse(object="list", model="private-gpt", data=splits)
