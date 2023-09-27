"""Tests for private-gpt."""
from unittest import TestCase
from unittest.mock import Mock

from injector import T

from private_gpt.di import root_injector


class BaseTestCase(TestCase):
    def setUp(self):
        self.test_injector = root_injector.create_child_injector()

    def inject_mock(self, interface: type[T]) -> T:
        mock_instance = Mock()
        self.test_injector.binder.bind(interface, to=mock_instance)
        return mock_instance

    def get(self, interface: type[T]) -> T:
        return self.test_injector.get(interface)
