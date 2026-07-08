# Celery
import logging
from typing import TYPE_CHECKING, Any

from celery import states
from pydantic import BaseModel

from private_gpt.celery import states as custom_states
from private_gpt.celery.error import CeleryError
from private_gpt.components.broker.broker_component import BrokerComponent
from private_gpt.di import get_global_injector
from private_gpt.server.utils.callback import AsyncResponse, BaseCallbackInput, Callback

if TYPE_CHECKING:
    from private_gpt.server.utils.callback import AMQP


logger = logging.getLogger(__name__)


def _publish_callback(
    exchange: str,
    routing_key: str,
    async_response: AsyncResponse,
    final: bool = True,
) -> None:
    broker_component = get_global_injector().get(BrokerComponent)
    broker_component.publish(
        exchange=exchange,
        routing_key=routing_key,
        body=bytes(async_response.model_dump_json(), "utf-8"),
    )
    if final:
        logger.debug(
            f"Published final callback message to {exchange}/{routing_key}: {async_response}"
        )
        broker_component.join()


def run_callback(
    task: Any,
    state: str,
    result: BaseModel,
    callback: Callback,
) -> None:
    callback_amqp: AMQP = callback.amqp
    final = False

    if state == states.SUCCESS:
        async_response = AsyncResponse(
            data=result,
            type=f"pgpt.{task.name}.done",
            error=None,
            callback_properties=callback.properties,
        )
        routing_key = callback_amqp.routing_key_done or async_response.type
        final = True
    elif state == custom_states.PROGRESS:
        async_response = AsyncResponse(
            data=result.model_dump(),
            type=f"pgpt.{task.name}.progress",
            error=None,
            callback_properties=callback.properties,
        )
        routing_key = callback_amqp.routing_key_progress or async_response.type
    else:
        # Unify all errors as CeleryError
        error_result = (
            result
            if isinstance(result, CeleryError)
            else CeleryError(errors=[str(result)])
        )
        logger.error(
            f"Task {task.name} failed with state {error_result.details.errors}",
            exc_info=error_result,
        )

        async_response = AsyncResponse(
            data=None,
            type=f"pgpt.{task.name}.error",
            error=error_result.dict(),
            callback_properties=callback.properties,
        )
        routing_key = callback_amqp.routing_key_error or async_response.type
        final = True

    _publish_callback(
        exchange=callback_amqp.exchange,
        routing_key=routing_key,
        async_response=async_response,
        final=final,
    )


def task_after_return(
    task: Any,
    state: str,
    result: BaseModel,
    _task_id: str,
    args: Any,
    _kwargs: dict[str, Any],
    _none: Any,
) -> None:
    """Callback After any task is completed.

    Every task will produce a message back to the broker
    with the result of the task. The payload will have a special
    "type" field with shape `pgpt.{task_name}.done` that will indicate
    The type of the data that is being sent.

    In case of error a suffix ".error" will be added to the type field.
    """
    # If the input is not a subclass of BaseCallbackInput, then we don't need to
    # send a callback
    if not args or not issubclass(args[0].__class__, BaseCallbackInput):
        return

    # If the input is a subclass of BaseCallbackInput, it needs to be the only input
    assert len(args) == 1, "Tasks must have only one argument, which must be the input"

    # Get callback arguments from the task args
    if args[0].callback is None:
        return

    # Run the callback defined in the task
    callback: Callback = args[0].callback
    run_callback(task, state, result, callback)
