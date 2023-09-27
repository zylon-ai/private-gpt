# TODO: Delete this later
from random import random

from injector import inject


class C:
    def __init__(self):
        self.number = random()


class B:
    @inject
    def __init__(self, c: C):
        self.message = "hi " + str(c.number)


class A:
    @inject
    def __init__(self, b: B):
        self.b = b

    def print(self):
        print(self.b.message)

    def message(self):
        return self.b.message
