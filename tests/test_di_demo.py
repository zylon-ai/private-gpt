# TODO: Delete this later
from tests import BaseTestCase
from tests.di_demo import A, C


class DiDemoTest(BaseTestCase):
    def test_hello(self):
        mock_c = self.inject_mock(C)
        mock_c.number = 3
        a = self.get(A)
        assert a.message() == "hi 3"

    def test_hello2(self):
        a = self.get(A)
        assert a.message().startswith("hi")
