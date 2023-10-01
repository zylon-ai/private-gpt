from private_gpt.completions.completions_service import CompletionsService
from tests.fixtures.mock_injector import MockInjector


def test_completions_service_produces_a_stream(injector: MockInjector) -> None:
    service = injector.get(CompletionsService)
    response = service.stream_complete("test", model="mock")
    text = "".join([r.delta or "" for r in response])
    assert text == "test"
