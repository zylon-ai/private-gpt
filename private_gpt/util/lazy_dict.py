import threading
from typing import Generic

from private_gpt.typing import K, V


class LazyDict(Generic[K, V]):
    """A lazy dictionary, thread-safe (but not very efficient).

    Lambda values will be evaluated only once lazily.
    """

    delegate: dict[K, V]

    def __init__(self, delegate: dict[K, V]):
        self.lock = threading.Lock()
        self.delegate = delegate

    def __getitem__(self, k: K) -> V:
        v = self.delegate[k]
        if callable(v):
            with self.lock:
                # second check inside the lock
                v = self.delegate[k]
                if callable(v):
                    # Compute only once
                    v = v()
                    self.delegate[k] = v
        return v
