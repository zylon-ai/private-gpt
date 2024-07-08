import datetime
import logging
import math
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


def human_time(*args: Any, **kwargs: Any) -> str:
    def timedelta_total_seconds(timedelta: datetime.timedelta) -> float:
        return (
            timedelta.microseconds
            + 0.0
            + (timedelta.seconds + timedelta.days * 24 * 3600) * 10**6
        ) / 10**6

    secs = float(timedelta_total_seconds(datetime.timedelta(*args, **kwargs)))
    # We want (ms) precision below 2 seconds
    if secs < 2:
        return f"{secs * 1000}ms"
    units = [("y", 86400 * 365), ("d", 86400), ("h", 3600), ("m", 60), ("s", 1)]
    parts = []
    for unit, mul in units:
        if secs / mul >= 1 or mul == 1:
            if mul > 1:
                n = int(math.floor(secs / mul))
                secs -= n * mul
            else:
                # >2s we drop the (ms) component.
                n = int(secs)
            if n:
                parts.append(f"{n}{unit}")
    return " ".join(parts)


def eta(iterator: list[Any]) -> Any:
    """Report an ETA after 30s and every 60s thereafter."""
    total = len(iterator)
    _eta = ETA(total)
    _eta.needReport(30)
    for processed, data in enumerate(iterator, start=1):
        yield data
        _eta.update(processed)
        if _eta.needReport(60):
            logger.info(f"{processed}/{total} - ETA {_eta.human_time()}")


class ETA:
    """Predict how long something will take to complete."""

    def __init__(self, total: int):
        self.total: int = total  # Total expected records.
        self.rate: float = 0.0  # per second
        self._timing_data: deque[tuple[float, int]] = deque(maxlen=100)
        self.secondsLeft: float = 0.0
        self.nexttime: float = 0.0

    def human_time(self) -> str:
        if self._calc():
            return f"{human_time(seconds=self.secondsLeft)} @ {int(self.rate * 60)}/min"
        return "(computing)"

    def update(self, count: int) -> None:
        # count should be in the range 0 to self.total
        assert count > 0
        assert count <= self.total
        self._timing_data.append((time.time(), count))  # (X,Y) for pearson

    def needReport(self, whenSecs: int) -> bool:
        now = time.time()
        if now > self.nexttime:
            self.nexttime = now + whenSecs
            return True
        return False

    def _calc(self) -> bool:
        # A sample before a prediction.   Need two points to compute slope!
        if len(self._timing_data) < 3:
            return False

        # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
        # Calculate means and standard deviations.
        samples = len(self._timing_data)
        # column wise sum of the timing tuples to compute their mean.
        mean_x, mean_y = (
            sum(i) / samples for i in zip(*self._timing_data, strict=False)
        )
        std_x = math.sqrt(
            sum(pow(i[0] - mean_x, 2) for i in self._timing_data) / (samples - 1)
        )
        std_y = math.sqrt(
            sum(pow(i[1] - mean_y, 2) for i in self._timing_data) / (samples - 1)
        )

        # Calculate coefficient.
        sum_xy, sum_sq_v_x, sum_sq_v_y = 0.0, 0.0, 0
        for x, y in self._timing_data:
            x -= mean_x
            y -= mean_y
            sum_xy += x * y
            sum_sq_v_x += pow(x, 2)
            sum_sq_v_y += pow(y, 2)
        pearson_r = sum_xy / math.sqrt(sum_sq_v_x * sum_sq_v_y)

        # Calculate regression line.
        # y = mx + b where m is the slope and b is the y-intercept.
        m = self.rate = pearson_r * (std_y / std_x)
        y = self.total
        b = mean_y - m * mean_x
        x = (y - b) / m

        # Calculate fitted line (transformed/shifted regression line horizontally).
        fitted_b = self._timing_data[-1][1] - (m * self._timing_data[-1][0])
        fitted_x = (y - fitted_b) / m
        _, count = self._timing_data[-1]  # adjust last data point progress count
        adjusted_x = ((fitted_x - x) * (count / self.total)) + x
        eta_epoch = adjusted_x

        self.secondsLeft = max([eta_epoch - time.time(), 0])
        return True
