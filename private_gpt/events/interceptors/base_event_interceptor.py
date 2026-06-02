from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from private_gpt.events.models import Event


class BaseEventInterceptor(ABC):
    @abstractmethod
    async def intercept(
        self, gen: AsyncGenerator[Event, None]
    ) -> AsyncGenerator[Event, None]:
        """Intercepts events from an async generator.

        Args:
            gen: An async generator yielding Event objects.

        Yields:
            Event: The intercepted Event objects.
        """
        pass
