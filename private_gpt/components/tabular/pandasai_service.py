import asyncio
import logging
from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol, cast

import pandas as pd
from pandasai import (  # type: ignore
    ConfigManager,
    DataFrame,
    Sandbox,
    VirtualDataFrame,
)
from pandasai.core.response import (  # type: ignore
    BaseResponse,
    ChartResponse,
    ErrorResponse,
)
from PIL.Image import Image
from pydantic import BaseModel, ConfigDict

from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.sandbox.sandbox_component import SandboxComponent
from private_gpt.components.tabular.pandasai_model import PGPTPandasAILLM
from private_gpt.components.tabular.pandasai_sandbox import PandasAISandboxAdapter
from private_gpt.server.principal import Principal
from private_gpt.settings.settings import settings
from private_gpt.utils.dataframe import df_to_minimal_markdown

if TYPE_CHECKING:
    from collections.abc import Callable

from injector import inject, singleton

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def format_number(value: int | float | None, decimals: int = 2) -> str:
    """Format number with thousand separators and appropriate decimal places."""
    if value is None:
        return "N/A"

    if isinstance(value, int) or value.is_integer():
        return f"{int(value):,}"

    return f"{value:,.{decimals}f}"


class PandasAIConfig(BaseModel):
    """Configuration for PandasAI."""

    max_retries: int = 3
    verbose: bool = settings().server.debug_mode
    seed: int = 0
    save_logs: bool = False


class PandaAIProtocol(Protocol):
    config: ConfigManager

    def create(
        self,
        path: str,
        df: pd.DataFrame | None = None,
        description: str | None = None,
        columns: list[dict[str, Any]] | None = None,
        source: dict[str, Any] | None = None,
        relations: list[dict[str, Any]] | None = None,
        view: bool = False,
        group_by: list[str] | None = None,
        transformations: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        ...

    def chat(
        self, query: str, *dataframes: pd.DataFrame, sandbox: Sandbox | None = None
    ) -> Any:
        ...

    def follow_up(self, query: str) -> Any:
        ...

    def load(self, dataset_path: str) -> pd.DataFrame:
        ...

    def read_csv(self, filepath: str) -> pd.DataFrame:
        ...


class PandasAIOutput(BaseModel):
    """Output of PandasAI."""

    response: BaseResponse
    error: str | None = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @property
    def value(self) -> Any:
        """Get the value of the response."""
        return self.response.value

    @property
    def content(self) -> list[Any]:
        """Get both raw and formatted content."""
        potential_result: list[Any | None] = (
            [self.error]
            if self.error
            else [ctr for ctr in (self.get_raw(), str(self)) if ctr]
        )

        hashable_types = (str, int, float, complex)
        hashable_result = [
            item for item in potential_result if isinstance(item, hashable_types)
        ]
        non_hashable_result = [
            item for item in potential_result if not isinstance(item, hashable_types)
        ]
        results = list(set(hashable_result)) + non_hashable_result
        if not results:
            self.error = "No content available"
        return results

    @property
    def last_code_executed(self) -> str | None:
        """Get the last code executed by the LLM."""
        last_code_executed = self.response.last_code_executed
        return str(last_code_executed) if last_code_executed else None

    def is_string(self) -> bool:
        """Check if the value is a string or integer."""
        return not self.is_number() and isinstance(self.value, str)

    def is_number(self) -> bool:
        """Check if the value is a number."""
        # Try to cast to int, float, or complex
        return isinstance(self.value, int | float | complex) and not isinstance(
            self.value, bool
        )

    def is_dataframe(self) -> bool:
        """Check if the value is a pandas DataFrame."""
        return isinstance(self.value, pd.DataFrame)

    def is_chart(self) -> bool:
        """Check if the response is a chart."""
        return isinstance(self.response, ChartResponse)

    def get_dataframe(self) -> pd.DataFrame:
        """Get the value as a DataFrame, or raise an error."""
        if not self.is_dataframe():
            raise ValueError("Output is not a pd.DataFrame")
        return cast(pd.DataFrame, self.value)

    def get_chart(self) -> Image:
        """Get the chart as an Image, or raise an error."""
        if not self.is_chart():
            raise ValueError("Output is not a chart")
        img = cast(ChartResponse, self.response)._get_image()
        return cast(Image, img)

    def _determine_response_type(self) -> str | None:
        """Determine the response type for dispatching."""
        type_checks = [
            ("dataframe", self.is_dataframe),
            ("chart", self.is_chart),
            ("string", self.is_string),
            ("number", self.is_number),
        ]

        for type_name, check_func in type_checks:
            if check_func():
                return type_name

        return None

    def get_raw(self) -> Any | None:
        """Get the raw output value in its appropriate format."""
        handlers: dict[str, Callable[[], Any]] = {
            "dataframe": lambda: None,
            "chart": self.get_chart,
            "string": lambda: str(self.value),
            "number": lambda: None,
        }

        response_type = self._determine_response_type()
        if response_type in handlers:
            return handlers[response_type]()

        raise ValueError("Output is not a string, number, pd.DataFrame, or chart")

    def __str__(self) -> str:
        """Convert the output to a string representation."""
        str_converters: dict[str, Callable[[], str]] = {
            "dataframe": lambda: df_to_minimal_markdown(self.get_dataframe())
            or "No results.",
            "chart": lambda: "Generated chart successfully. Plot was attached to the conversation."
            "Don't create placeholders for charts, just reply that the chart was generated.",
            "string": lambda: str(self.value),
            "number": lambda: format_number(self.value),
        }

        response_type = self._determine_response_type()
        if response_type and response_type in str_converters:
            return str_converters[response_type]()

        raise ValueError("Output is not a string, number, pd.DataFrame, or chart")


@singleton
class PandasAIService(BaseModel):
    _pandas_ai: PandaAIProtocol

    _llm: PGPTPandasAILLM
    _config: PandasAIConfig
    _sandbox_component: SandboxComponent

    @inject
    def __init__(
        self, llm_component: LLMComponent, sandbox_component: SandboxComponent
    ) -> None:
        super().__init__()
        self._pandas_ai = import_module("pandasai")  # type: ignore
        self._llm = PGPTPandasAILLM(
            llm=llm_component.llm, llm_alias=llm_component.alias
        )
        self._config = PandasAIConfig()
        self._configure(self._llm, self._config)
        self._sandbox_component = sandbox_component

    @property
    def pai(self) -> PandaAIProtocol:
        """Get the underlying PandasAI instance.

        This instance MUST be used to have the llm / sandbox configured properly.
        Otherwise, you will see an error like:
        Organization name must be lowercase and use hyphens instead of spaces
        """
        return self._pandas_ai

    def _configure(
        self,
        llm: PGPTPandasAILLM,
        config: PandasAIConfig,
    ) -> None:
        # Configure the PandasAI instance
        self._pandas_ai.config.set(
            {
                "llm": llm,
                "organization": "zylon",
                **config.dict(exclude_none=True),
            }
        )

        # Configure the logging
        logger_pandasai = logging.getLogger("pandasai")
        logger_pandasai.setLevel(logging.DEBUG if config.verbose else logging.WARNING)

    def _build_smart_dataframes(
        self, dataframes: tuple[pd.DataFrame, ...]
    ) -> list[DataFrame | VirtualDataFrame]:
        """Wrap plain pandas dataframes while preserving virtual dataframes."""
        return [
            dataframe
            if isinstance(dataframe, VirtualDataFrame)
            else DataFrame(dataframe)
            for dataframe in dataframes
        ]

    async def _resolve_sandbox(self, sandbox: Sandbox | None = None) -> Sandbox | None:
        """Resolve the sandbox used for execution.

        Priority:
        1. An explicit sandbox passed by the caller.
        2. A sandbox session created from the configured SandboxComponent.
        3. No sandbox when the feature is disabled.
        """
        if sandbox is not None:
            return sandbox

        session = await self._sandbox_component.create_session(
            env=Principal.current().as_env() or None,
        )
        if session is None:
            return None

        return PandasAISandboxAdapter(client=session)

    def _execute_chat(
        self,
        query: str,
        smart_dataframes: list[DataFrame | VirtualDataFrame],
        sandbox: Sandbox | None,
    ) -> BaseResponse:
        """Execute the PandasAI chat call, managing sandbox lifecycle if needed."""
        try:
            if sandbox is not None:
                sandbox.start()

            result = self._pandas_ai.chat(query, *smart_dataframes, sandbox=sandbox)
            if not result:
                raise ValueError("We don't have any result")
            return result
        finally:
            if sandbox is not None:
                sandbox.stop()

    def _run_analysis_sync(
        self,
        query: str,
        smart_dataframes: list[DataFrame | VirtualDataFrame],
        sandbox: Sandbox | None,
    ) -> PandasAIOutput:
        """Synchronous analysis execution with sandbox management."""
        try:
            result = self._execute_chat(query, smart_dataframes, sandbox)
        except Exception as e:
            logger.error(f"Error during PandasAI analysis: {e}")
            result = ErrorResponse(error=str(e))

        return PandasAIOutput(
            response=result,
            error=result.value if isinstance(result, ErrorResponse) else None,
        )

    async def run_analysis(
        self,
        query: str,
        *dataframes: pd.DataFrame,
        **kwargs: Any,
    ) -> PandasAIOutput:
        smart_dataframes = self._build_smart_dataframes(dataframes)
        sandbox = await self._resolve_sandbox(kwargs.get("sandbox"))

        logger.debug(
            f"Running analysis with query: '{query}' using {len(smart_dataframes)} dataframes"
            + (f" over sandbox: {sandbox.__class__.__name__}" if sandbox else "")
        )

        result = await asyncio.to_thread(
            self._run_analysis_sync, query, smart_dataframes, sandbox
        )

        logger.debug(
            f"Result: {result.response} "
            f"Type: {type(result.response)} "
            f"Last code executed: {result.response.last_code_executed}"
        )

        if sandbox is not None:
            await asyncio.to_thread(sandbox.stop)

        return result
