# create a enum with different priorities. Lower is higher
from enum import IntFlag


class Priority(IntFlag):
    DEFAULT = 0
    REAL_TIME = 1
    NO_PRIORITY = 2


class DefinedPriorities:
    """Class to define and group all priorities for LLM and Embedding."""

    class LLM:
        # Default priority for LLM
        DEFAULT_PRIORITY = Priority.NO_PRIORITY

        # Different priorities for different use cases
        CHAT_PRIORITY = Priority.REAL_TIME
        SUMMARY_PRIORITY = Priority.NO_PRIORITY
        REPORT_PRIORITY = Priority.NO_PRIORITY

    class Embedding:
        # Default priority for Embedding
        DEFAULT_PRIORITY = Priority.DEFAULT
