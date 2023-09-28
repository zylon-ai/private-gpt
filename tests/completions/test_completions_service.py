from private_gpt.completions.completions_service import CompletionsService
from tests.common import BaseTestCase


class TestCompletionsService(BaseTestCase):
    def test_completions_service_produces_an_stream(self):
        service = self.get(CompletionsService)
        response = service.stream_complete("test", model="mock")
        deltas = []
        for r in response:
            deltas.append(r.delta)

        assert deltas == ["t", "e", "s", "t"]
