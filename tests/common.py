from collections.abc import Callable
from unittest import TestCase
from unittest.mock import MagicMock

from injector import Provider, ScopeDecorator, singleton

from private_gpt.di import create_application_injector
from private_gpt.typing import T


class BaseTestCase(TestCase):
    def setUp(self):
        self.test_injector = create_application_injector()

    def bind_mock(
        self,
        interface: type[T],
        mock: (T | (Callable[..., T] | Provider[T])) | None = None,
        *,
        scope: ScopeDecorator = singleton,
    ) -> T:
        if mock is None:
            mock = MagicMock()
        self.test_injector.binder.bind(interface, to=mock, scope=scope)
        return mock

    def get(self, interface: type[T]) -> T:
        return self.test_injector.get(interface)
