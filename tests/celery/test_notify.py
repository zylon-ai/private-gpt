import itertools
import time
from unittest.mock import Mock

import pytest

from private_gpt.celery.notify import ProgressStatus, ProgressStep, notify_progress

PRECISION = 1e-3


class MockProgressStep(ProgressStep):
    STEP_ONE = "step_one"
    STEP_TWO = "step_two"
    STEP_THREE = "step_three"


def test_progress_status_from_percentage():
    # Set the current step for the test
    ProgressStatus.current_step = MockProgressStep.STEP_TWO

    # Test from_percentage method with 50% in the second step
    result = ProgressStatus.from_percentage(percentage=50)

    # Calculate the expected percentage:
    expected_percentage = 50
    assert result.percentage == pytest.approx(expected_percentage, rel=PRECISION), (
        f"The percentage should be close to {expected_percentage}, but got {result.percentage}."
    )


def test_calculate_percentage_out_of_bounds():
    # Test with an out-of-bounds percentage (>100%)
    provided_percentage = 150.0  # More than 100%

    # Call the static method
    result = ProgressStatus.from_percentage(provided_percentage)

    # The result should still cap at the maximum percentage for STEP_ONE (33.333%)
    expected_percentage = 100
    assert result.percentage == pytest.approx(expected_percentage, rel=PRECISION), (
        f"Expected capped percentage to be close to {expected_percentage} for STEP_ONE, but got {result}."
    )


def test_notify_progress():
    # Mock the notify function
    notify_mock = Mock()

    # Create a status class for testing
    class TestProgressStatus(ProgressStatus):
        current_step = MockProgressStep.STEP_ONE

    # Use the context manager
    with notify_progress(
        notify=notify_mock,
        status_class=TestProgressStatus,
        start_percentage=10,
        end_percentage=90,
    ):
        pass  # Simulating the work being done

    # Ensure notify was called at start and end
    assert notify_mock.call_count == 2, (
        "Notify should be called twice, once at start and once at end."
    )

    # Check that the correct progress was notified at the start
    notify_mock.assert_any_call(TestProgressStatus.from_percentage(percentage=10))

    # Check that the correct progress was notified at the end
    notify_mock.assert_any_call(
        TestProgressStatus.from_percentage(percentage=90, warnings=[])
    )


def test_notify_progress_with_warnings():
    # Mock the notify function
    notify_mock = Mock()

    # Create a status class for testing
    class TestProgressStatus(ProgressStatus):
        current_step = MockProgressStep.STEP_ONE

    # Define some warnings
    warnings = ["Warning 1", "Warning 2"]

    # Use the context manager with warnings
    with notify_progress(
        notify=notify_mock,
        status_class=TestProgressStatus,
        start_percentage=20,
        end_percentage=100,
    ) as progress:
        progress(warnings=warnings)

    # Check notify was called twice
    assert notify_mock.call_count == 3, "Notify should be called three with warnings."


@pytest.mark.parametrize(
    ("step", "provided_percentage", "expected_result"),
    [
        (
            MockProgressStep.STEP_ONE,
            0,
            0,
        ),  # 0% into STEP_ONE should give 0% total progress
        (
            MockProgressStep.STEP_ONE,
            50,
            50,
        ),  # 50% into STEP_ONE should give 50% total progress
        (
            MockProgressStep.STEP_ONE,
            100,
            100,
        ),  # 100% into STEP_ONE should give 100% total progress
        (
            MockProgressStep.STEP_TWO,
            0,
            0,
        ),  # 0% into STEP_ONE should give 0% total progress
        (
            MockProgressStep.STEP_TWO,
            50,
            50,
        ),  # 50% into STEP_ONE should give 50% total progress
        (
            MockProgressStep.STEP_TWO,
            100,
            100,
        ),  # 100% into STEP_ONE should give 100% total progress
        (
            MockProgressStep.STEP_THREE,
            0,
            0,
        ),  # 0% into STEP_ONE should give 0% total progress
        (
            MockProgressStep.STEP_THREE,
            50,
            50,
        ),  # 50% into STEP_ONE should give 50% total progress
        (
            MockProgressStep.STEP_THREE,
            100,
            100,
        ),  # 100% into STEP_ONE should give 100% total progress
    ],
)
def test_progress_status_boundary_cases(
    step: MockProgressStep, provided_percentage: float, expected_result: float
):
    # Set current step for testing
    ProgressStatus.current_step = step

    # Test from_percentage method with boundary cases
    result = ProgressStatus.from_percentage(percentage=provided_percentage)

    # Ensure the percentage matches the expected result with a small tolerance
    assert result.percentage == pytest.approx(expected_result, rel=PRECISION), (
        f"Expected {expected_result}%, but got {result.percentage}%."
    )


def test_fake_generator_progress() -> None:
    notifications: list[float] = []
    start_time = time.time()

    # Create a status class for testing
    class TestProgressStatus(ProgressStatus):
        current_step = MockProgressStep.STEP_ONE

    def notify(status: ProgressStatus) -> None:
        notifications.append(status.percentage or 0.0)

    with notify_progress(
        notify=notify,
        status_class=TestProgressStatus,
        generate_fake_percentage=True,
        generate_fake_percentage_interval_ms=100,
        generate_fake_percentage_jitter=10,
    ):
        # Simulate work being done
        time.sleep(0.5)

    end_time = time.time()
    duration = end_time - start_time

    assert len(notifications) > 3
    assert notifications[0] == 0.0
    assert notifications[-1] == pytest.approx(100, rel=PRECISION), (
        f"Expected 100%, but got {notifications[-1]}%."
    )
    assert duration < 1.0
    assert all(x <= y for x, y in itertools.pairwise(notifications))
