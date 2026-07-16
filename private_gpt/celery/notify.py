import enum
import math
from abc import ABC
from collections.abc import Callable, Generator
from contextlib import contextmanager
from threading import Event, Thread
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel


class ProgressStep(enum.StrEnum):
    pass


class ProgressStatus(BaseModel, ABC):
    current_step: ClassVar[ProgressStep]
    percentage: float | None = None
    warnings: list[str] | None = None

    @classmethod
    def from_percentage(
        cls: type["ProgressStatus"],
        percentage: float,
        warnings: list[str] | None = None,
    ) -> "ProgressStatus":
        """Create a ProgressStatus from a percentage."""
        return cls(
            percentage=min(max(percentage, 0), 100),
            warnings=warnings,
        )

    def model_dump(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        original = super().model_dump(**kwargs)
        original["current_step"] = self.current_step
        return original

    def model_dump_json(
        self,
        **kwargs: Any,
    ) -> str:
        import json

        return json.dumps(self.model_dump(**kwargs))


class NotifyProtocol(Protocol):
    def __call__(
        self, percentage: float | None = None, warnings: list[str] | None = None
    ) -> None:
        """This method signature defines how notify should behave."""
        pass


def fake_percentage_generator(
    start_percentage: float,
    end_percentage: float,
    generate_fake_percentage_interval_ms: int,
    generate_fake_percentage_jitter: float,
    notify_custom: NotifyProtocol,
) -> Thread:
    """Generates fake percentage updates in a separate thread.

    Parameters:
    - start_percentage (float): The starting percentage.
    - end_percentage (float): The ending percentage.
    - generate_fake_percentage_interval_ms (int): Interval between updates.
    - generate_fake_percentage_jitter (float): Jitter to apply to percentage updates.
    - notify_custom (callable): Callback function to notify with the updated percentage.

    Returns:
    - threading.Thread: The thread running the fake percentage generator.
    """

    def generate_fake_percentage_fn() -> None:
        import time

        last_percentage = start_percentage
        while not stop_event.is_set():
            time.sleep(generate_fake_percentage_interval_ms / 1000)
            if last_percentage < end_percentage:
                percentage = last_percentage + generate_fake_percentage_jitter
                percentage = min(max(percentage, start_percentage), end_percentage)
                percentage = math.floor(percentage * 100) / 100

                if percentage != last_percentage:
                    notify_custom(percentage=percentage)
                    last_percentage = percentage
            else:
                notify_custom(percentage=end_percentage)
                break

    stop_event = Event()
    fake_generator_thread = Thread(target=generate_fake_percentage_fn)
    fake_generator_thread.start()

    def stop_generator() -> None:
        stop_event.set()
        fake_generator_thread.join()

    fake_generator_thread.stop = stop_generator  # type: ignore
    return fake_generator_thread


@contextmanager
def notify_progress(
    notify: Callable[[ProgressStatus], None],
    status_class: type[ProgressStatus],
    start_percentage: float = 0,
    end_percentage: float = 100,
    warnings: list[str] | None = None,
    generate_fake_percentage: bool = False,
    generate_fake_percentage_interval_ms: int | None = None,
    generate_fake_percentage_jitter: float | None = None,
) -> Generator[NotifyProtocol, None, None]:
    """Context manager to notify progress status."""
    # Notify the start
    notify(status_class.from_percentage(percentage=start_percentage, warnings=warnings))

    # Yield decorated code
    last_percentage: float = start_percentage
    last_warnings: list[str] = warnings or []

    def notify_custom(
        percentage: float | None = None, warnings: list[str] | None = None
    ) -> None:
        nonlocal last_percentage, last_warnings

        percentage = percentage or last_percentage
        new_warnings = list(set(last_warnings + (warnings or [])))

        if (percentage >= last_percentage) and (
            percentage != last_percentage or new_warnings != last_warnings
        ):
            last_percentage = percentage
            last_warnings = new_warnings
            notify(
                status_class.from_percentage(
                    percentage=percentage, warnings=last_warnings
                )
            )

    fake_generator: Thread | None = None
    if generate_fake_percentage:
        fake_generator = fake_percentage_generator(
            start_percentage=start_percentage,
            end_percentage=end_percentage - 5,  # Avoid reaching 100%
            generate_fake_percentage_interval_ms=generate_fake_percentage_interval_ms
            or 1000,
            generate_fake_percentage_jitter=generate_fake_percentage_jitter or 1,
            notify_custom=notify_custom,
        )

    success = False
    try:
        yield notify_custom
        success = True
    finally:
        stop_generator = getattr(fake_generator, "stop", None)
        if callable(stop_generator):
            stop_generator()

        if success:
            # Notify the end
            notify(
                status_class.from_percentage(
                    percentage=end_percentage, warnings=last_warnings
                )
            )
