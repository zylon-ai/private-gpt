from typing import Any
from unittest.mock import Mock

from pydantic import BaseModel

from private_gpt.celery.callback import task_after_return
from private_gpt.celery.celery import celery_app
from private_gpt.celery.error import CeleryError
from private_gpt.components.broker.broker_component import BrokerComponent
from private_gpt.di import (
    set_global_injector,
)
from private_gpt.server.utils.callback import (
    AMQP,
    AsyncResponse,
    BaseCallbackInput,
    Callback,
)
from tests.fixtures.mock_injector import MockInjector

"""
Note: we are using the app's main celery instance. Any registered task in any test
remains registered for all tests. Don't override real tasks names or reuse names
among tests.
"""


class CallbackInput(BaseCallbackInput):
    x: int
    y: int


class CallbackResponse(BaseModel):
    result: int
    label: str


def test_success_task_posts_to_success_broker_queue(injector: MockInjector):
    broker_mock = Mock(BrokerComponent)
    injector.bind_mock(BrokerComponent, broker_mock)

    set_global_injector(injector.test_injector)

    @celery_app.task(name="mul_task", after_return=task_after_return)
    def mul(input_with_callback: CallbackInput) -> CallbackResponse:
        return CallbackResponse(
            result=input_with_callback.x * input_with_callback.y,
            label="test",
        )

    celery_app.send_task(
        "mul_task",
        args=(
            CallbackInput(
                x=2,
                y=3,
                callback=Callback(
                    amqp=AMQP(
                        exchange="main",
                        routing_key_done="mul.done",
                        routing_key_progress="mul.progress",
                        routing_key_error="mul.error",
                    ),
                    properties={"test": "123"},
                ),
            ),
        ),
    )

    expected_response = AsyncResponse(
        data=CallbackResponse(
            result=6, label="test", callback_properties={"test": "123"}
        ),
        type="pgpt.mul_task.done",
        callback_properties={"test": "123"},
    )

    broker_mock.publish.assert_called_once_with(
        exchange="main",
        routing_key="mul.done",
        body=bytes(expected_response.model_dump_json(), "utf-8"),
    )


def test_failing_task_posts_to_error_handler_queue(injector: MockInjector):
    broker_mock = Mock(BrokerComponent)
    injector.bind_mock(BrokerComponent, broker_mock)

    set_global_injector(injector.test_injector)

    @celery_app.task(name="err_task", after_return=task_after_return)
    def mul(_input_with_callback: CallbackInput) -> Any:
        raise Exception("Test exception")

    celery_app.send_task(
        "err_task",
        args=(
            CallbackInput(
                x=2,
                y=3,
                callback=Callback(
                    amqp=AMQP(
                        exchange="main",
                        routing_key_done="mul.done",
                        routing_key_progress="mul.progress",
                        routing_key_error="mul.error",
                    ),
                    properties={"test": "123"},
                ),
            ),
        ),
    )

    expected_response = AsyncResponse(
        data=None,
        error=CeleryError(errors=[str(Exception("Test exception"))]).dict(),
        callback_properties={"test": "123"},
        type="pgpt.err_task.error",
    )

    broker_mock.publish.assert_called_once_with(
        exchange="main",
        routing_key="mul.error",
        body=bytes(expected_response.model_dump_json(), "utf-8"),
    )


def test_success_task_without_callback(injector: MockInjector):
    broker_mock = Mock(BrokerComponent)
    injector.bind_mock(BrokerComponent, broker_mock)

    set_global_injector(injector.test_injector)

    @celery_app.task(name="mul_task", after_return=task_after_return)
    def mul(input_with_callback: CallbackInput) -> CallbackResponse:
        return CallbackResponse(
            result=input_with_callback.x * input_with_callback.y, label="test"
        )

    celery_app.send_task(
        "mul_task",
        args=(
            CallbackInput(
                x=2,
                y=3,
            ),
        ),
    )

    broker_mock.publish.assert_not_called()
