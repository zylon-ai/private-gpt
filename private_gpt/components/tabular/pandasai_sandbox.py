import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import textwrap
import uuid
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import pandas as pd
from pandasai import Sandbox  # type: ignore
from pandasai.exceptions import CodeExecutionError  # type: ignore

from private_gpt.components.sandbox.base import (
    SandboxCodeOptions,
    SandboxExecutionResult,
    SandboxSession,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text-cleaning helpers
# ---------------------------------------------------------------------------


def clean_exception_text(text: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    cleaned = ansi_escape.sub("", text)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
    cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
    return cleaned.strip()


def clean_traceback_string(tb_string: str) -> str:
    lines = tb_string.split("\n")
    cleaned_lines = [
        clean_exception_text(line)
        for line in lines
        if clean_exception_text(line).strip()
    ]
    return "\n".join(cleaned_lines)


def get_clean_exception_info(exc: Exception) -> str:
    return f"{type(exc).__name__}: {clean_exception_text(str(exc))}"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class PandasAISandboxAdapter(Sandbox):  # type: ignore[misc]

    _user_id: str
    _timeout: int
    _client: SandboxSession | None
    _loop: asyncio.AbstractEventLoop | None
    _temp_dir: Path | None
    _started: bool

    # Preamble injected at the top of EVERY run_code call so that each
    # exec context is fully self-contained and no session globals are needed.
    _PREAMBLE: str = textwrap.dedent(
        """
        import os
        import sys
        import json
        import uuid
        import base64
        import datetime
        import io
        from json import JSONEncoder
        from typing import Any

        import numpy as np
        import pandas as pd

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    """
    ).strip()

    _CUSTOM_COLORS: ClassVar[list[str]] = [
        "#7768BF",
        "#7A7E84",
        "#B5B7BA",
        "#E1E2E4",
        "#CA62A7",
        "#4E96BF",
        "#BBB3DF",
        "#E5B0D3",
        "#A7CBDF",
        "#E4E1F2",
        "#F4E0ED",
        "#DCEAF2",
    ]

    def __init__(
        self,
        user_id: str | None = None,
        timeout: int = 60,
        client: SandboxSession | None = None,
    ) -> None:
        super().__init__()
        # Force Agg for the host process too, so any accidental pyplot import
        # on the main thread doesn't trigger the macOS NSWindow crash.
        os.environ.setdefault("MPLBACKEND", "Agg")

        self._user_id = user_id or f"zylon_sandbox_{uuid.uuid4().hex[:8]}"
        self._timeout = timeout
        self._client = client
        self._temp_dir = None
        self._started = False

        # Capture the event loop the async sandbox session belongs to. PandasAI
        # drives this adapter from a worker thread (asyncio.to_thread), so async
        # sandbox calls are submitted back to this loop via _run().
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    # ------------------------------------------------------------------
    # Sync → async bridge
    # ------------------------------------------------------------------

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run an async sandbox call from this adapter's sync interface.

        Must not be called from the captured loop's own thread — it blocks
        until the coroutine completes.
        """
        if self._loop is not None and self._loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, self._loop).result()
        return asyncio.run(coro)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return

        logger.debug("Starting remote sandbox session for user: %s", self._user_id)

        try:
            if self._client is None:
                raise RuntimeError("Sandbox client not configured")
            self._temp_dir = Path(
                tempfile.mkdtemp(prefix=f"zylon_sandbox_{self._user_id}_")
            )
            self._setup_environment()
            self._started = True
            logger.debug("Remote sandbox session started successfully")
        except Exception as e:
            logger.error("Failed to start remote sandbox: %s", e)
            raise RuntimeError(f"Failed to start sandbox: {e}") from e

    def stop(self) -> None:
        if not self._started:
            return

        logger.debug("Stopping remote sandbox session for user: %s", self._user_id)

        try:
            if self._client:
                self._run(self._client.close())
            if self._temp_dir and self._temp_dir.exists():
                shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception as e:
            logger.error("Error stopping sandbox: %s", e)
        finally:
            self._started = False
            self._client = None
            self._temp_dir = None

        logger.debug("Remote sandbox session stopped successfully")

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------

    def _setup_environment(self) -> None:
        if not self._client:
            raise RuntimeError("Client not initialized")

        temp_dir = f"/tmp/{self._user_id}"
        colors_repr = repr(self._CUSTOM_COLORS)

        setup_code = textwrap.dedent(
            f"""
            {self._PREAMBLE}

            CUSTOM_COLORS = {colors_repr}

            plt.rcParams.update({{
                'figure.facecolor': 'none',
                'axes.facecolor': 'none',
                'savefig.transparent': True,
                'axes.edgecolor': 'none',
                'axes.linewidth': 0,
                'xtick.bottom': False,
                'ytick.left': False,
                'axes.grid': False,
                'axes.prop_cycle': plt.cycler('color', CUSTOM_COLORS),
            }})

            TEMP_DIR = {temp_dir!r}
            os.makedirs(TEMP_DIR, exist_ok=True)

            def clean_axes(ax):
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(
                    top=False, bottom=False, left=False, right=False,
                    labeltop=False, labelbottom=False,
                    labelleft=False, labelright=False,
                )

            def save_chart(fig, filename=None, clean_style=True):
                if filename is None:
                    filename = f"chart_{{uuid.uuid4().hex[:8]}}.png"
                filepath = os.path.join(TEMP_DIR, filename)
                if clean_style:
                    for ax in fig.get_axes():
                        clean_axes(ax)
                fig.savefig(filepath, dpi=300, bbox_inches='tight', transparent=True)
                plt.close(fig)
                return filepath

            def save_dataframe(df, filename=None, fmt='csv'):
                if filename is None:
                    filename = f"data_{{uuid.uuid4().hex[:8]}}.{{fmt}}"
                filepath = os.path.join(TEMP_DIR, filename)
                if fmt == 'csv':
                    df.to_csv(filepath, index=False)
                elif fmt == 'json':
                    df.to_json(filepath, orient='records', indent=2)
                elif fmt in ('xlsx', 'excel'):
                    df.to_excel(filepath, index=False)
                else:
                    raise ValueError(f"Unsupported format: {{fmt}}")
                return filepath
        """
        ).strip()

        try:
            result = self._run(
                self._client.run_code(setup_code, SandboxCodeOptions(language="python"))
            )
            if not result.success:
                logger.warning("Environment setup warning: %s", result.error)
        except Exception as e:
            logger.warning("Error during environment setup: %s", e)

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def _exec_code(self, code: str, environment: dict[str, Any]) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("Sandbox not started. Call start() first.")

        try:
            sql_queries = self._extract_sql_queries_from_code(code)
            datasets_code, exceptions = self._process_sql_queries(
                sql_queries, environment
            )
            if exceptions:
                raise ValueError(
                    f"Failed to execute some SQL queries: "
                    f"{', '.join(str(e) for e in exceptions)}"
                )

            processed_code = self._prepare_code_for_execution(code)
            full_code = "\n\n".join(
                part for part in (self._PREAMBLE, datasets_code, processed_code) if part
            )

            execution_result = self._run(
                self._client.run_code(
                    full_code,
                    SandboxCodeOptions(language="python", timeout=self._timeout),
                )
            )
            return self._process_execution_result(execution_result)

        except Exception as e:
            message = get_clean_exception_info(e)
            logger.debug("Code execution failed: %s", message)
            raise CodeExecutionError(f"Code execution failed: {message}")  # noqa: B904

    def _process_sql_queries(
        self, sql_queries: list[str], environment: dict[str, Any]
    ) -> tuple[str, list[Exception]]:
        if not sql_queries:
            return "", []

        temp_dir = f"/tmp/{self._user_id}"
        datasets_map: dict[str, str] = {}
        exceptions: list[Exception] = []

        for sql_query in sql_queries:
            execute_sql_query_func = environment.get("execute_sql_query")
            if execute_sql_query_func is None:
                logger.warning("execute_sql_query function not found in environment")
                continue

            try:
                query_df = execute_sql_query_func(sql_query)
                filename = f"query_result_{uuid.uuid4().hex[:8]}.csv"
                self.transfer_file(query_df, filename)
                datasets_map[sql_query] = filename
            except Exception as e:
                custom_exception = e.__class__(clean_exception_text(str(e)))
                exceptions.append(custom_exception)
                logger.error("Failed to execute SQL query: %s", custom_exception)

        if not datasets_map and exceptions:
            return "", exceptions

        datasets_code = textwrap.dedent(
            f"""
            import os
            import pandas as pd
            _datasets_map = {datasets_map!r}
            _temp_dir = {temp_dir!r}
            def execute_sql_query(sql_query):
                filename = _datasets_map.get(sql_query)
                if filename:
                    return pd.read_csv(os.path.join(_temp_dir, filename))
                raise ValueError(f'Query not found: {{sql_query}}')
        """
        ).strip()

        return datasets_code, []

    def _prepare_code_for_execution(self, code: str) -> str:
        temp_dir = f"/tmp/{self._user_id}"

        # Redirect any hardcoded .png paths into the sandbox temp dir
        code = re.sub(
            r"""(['"])([^'"]*\.png)\1""",
            lambda m: (
                f"{m.group(1)}{temp_dir}/{os.path.basename(m.group(2))}{m.group(1)}"
            ),
            code,
        )

        # Replace explicit color lists with CUSTOM_COLORS
        def color_replacer(match: re.Match[str]) -> str:
            param_name = match.group(1)
            color_count = len(re.findall(r"['\"][^'\"]*['\"]", match.group(0)))
            colors_repr = repr(self._CUSTOM_COLORS)
            if color_count <= 12:
                return f"{param_name}={colors_repr}[:{color_count}]"
            repeats = (color_count // 12) + 1
            return f"{param_name}=({colors_repr} * {repeats})[:{color_count}]"

        code = re.sub(
            r"(c|colors?)\s*=\s*\[\s*(?:['\"][^'\"]*['\"](?:\s*,\s*)?)+\s*\]",
            color_replacer,
            code,
        )

        # Append result serialization — all imports are explicit so this block
        # is self-contained regardless of what exec context it lands in.
        code += textwrap.dedent(
            """

            import os, json, base64, io, datetime
            from json import JSONEncoder
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

            class CustomEncoder(JSONEncoder):
                @staticmethod
                def serialize_dataframe(df: pd.DataFrame) -> dict:
                    if df.empty:
                        return {"columns": [], "data": [], "index": []}
                    return df.to_dict(orient="split")

                def default(self, obj):
                    if isinstance(obj, (np.integer, np.int64)):
                        return int(obj)
                    if isinstance(obj, (np.floating, np.float64)):
                        return float(obj)
                    if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
                        return obj.isoformat()
                    if isinstance(obj, pd.DataFrame):
                        return CustomEncoder.serialize_dataframe(obj)
                    return super().default(obj)

            _execution_result = locals().get('result', None)

            if isinstance(_execution_result, dict) and _execution_result.get('type') == 'plot':
                _chart_path = _execution_result.get('value')
                if _chart_path and os.path.exists(_chart_path):
                    with open(_chart_path, 'rb') as _f:
                        _image_b64 = base64.b64encode(_f.read()).decode('utf-8')
                    _execution_result['value'] = f"data:image/png;base64,{_image_b64}"
                else:
                    _buf = io.BytesIO()
                    plt.savefig(_buf, format='png', dpi=150, bbox_inches='tight')
                    _buf.seek(0)
                    _image_b64 = base64.b64encode(_buf.getvalue()).decode('utf-8')
                    _buf.close()
                    _execution_result['value'] = f"data:image/png;base64,{_image_b64}"

            print("EXECUTION_RESULT_START")
            print(json.dumps(_execution_result, cls=CustomEncoder))
            print("EXECUTION_RESULT_END")
        """
        )

        return code

    def _process_execution_result(
        self, result: SandboxExecutionResult
    ) -> dict[str, Any]:
        if not result.success:
            raise CodeExecutionError(
                f"Execution failed: {result.error or 'Unknown error'}"
            )

        try:
            lines = result.output.strip().split("\n")
            start_idx: int | None = None
            end_idx: int | None = None

            for i, line in enumerate(lines):
                if line.strip() == "EXECUTION_RESULT_START":
                    start_idx = i + 1
                elif line.strip() == "EXECUTION_RESULT_END":
                    end_idx = i
                    break

            if start_idx is not None and end_idx is not None:
                json_str = "\n".join(lines[start_idx:end_idx])
                json_obj: dict[str, Any] = json.loads(json_str)
                return self._convert_to_pandas_ai_response(json_obj)

            return {"type": "string", "value": str(result.output)}

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug("Could not parse execution result: %s", e)
            return {"type": "string", "value": str(result.output)}

    def _convert_to_pandas_ai_response(self, result: dict[str, Any]) -> dict[str, Any]:
        response_type = result.get("type", "string")
        response_value = result.get("value")

        if response_type == "dataframe" and isinstance(response_value, dict):
            response_value = pd.DataFrame(
                data=response_value["data"],
                index=response_value["index"],
                columns=response_value["columns"],
            )

        return {"type": response_type, "value": response_value}

    # ------------------------------------------------------------------
    # File operations — every generated snippet is self-contained
    # ------------------------------------------------------------------

    def transfer_file(self, csv_data: pd.DataFrame, filename: str = "file.csv") -> None:
        if not self._client:
            raise RuntimeError("Sandbox not started")

        temp_dir = f"/tmp/{self._user_id}"
        filepath = f"{temp_dir}/{filename}"
        csv_content = csv_data.to_csv(index=False)

        transfer_code = textwrap.dedent(
            f"""
            import os
            os.makedirs({temp_dir!r}, exist_ok=True)
            _csv_content = {csv_content!r}
            with open({filepath!r}, 'w') as _f:
                _f.write(_csv_content)
            print("File transferred: {filepath}")
        """
        ).strip()

        try:
            result = self._run(
                self._client.run_code(
                    transfer_code, SandboxCodeOptions(language="python")
                )
            )
            if not result.success:
                raise RuntimeError(f"File transfer failed: {result.error}")
            logger.debug("Successfully transferred file: %s", filename)
        except Exception as e:
            logger.error("Failed to transfer file %s: %s", filename, e)
            raise RuntimeError(f"File transfer failed: {e}") from e

    def get_file_content(self, filename: str) -> str | None:
        if not self._client:
            raise RuntimeError("Sandbox not started")

        temp_dir = f"/tmp/{self._user_id}"
        filepath = f"{temp_dir}/{filename}"

        read_code = textwrap.dedent(
            f"""
            import os
            _filepath = {filepath!r}
            if os.path.exists(_filepath):
                with open(_filepath, 'r') as _f:
                    _content = _f.read()
                print("FILE_CONTENT_START")
                print(_content)
                print("FILE_CONTENT_END")
            else:
                print(f"File not found: {{_filepath}}")
        """
        ).strip()

        try:
            result = self._run(
                self._client.run_code(read_code, SandboxCodeOptions(language="python"))
            )
            if result.success and "FILE_CONTENT_START" in result.output:
                lines = result.output.split("\n")
                start_idx: int | None = None
                end_idx: int | None = None
                for i, line in enumerate(lines):
                    if line.strip() == "FILE_CONTENT_START":
                        start_idx = i + 1
                    elif line.strip() == "FILE_CONTENT_END":
                        end_idx = i
                        break
                if start_idx is not None and end_idx is not None:
                    return "\n".join(lines[start_idx:end_idx])
        except Exception as e:
            logger.error("Failed to read file %s: %s", filename, e)

        return None

    def list_files(self) -> list[str]:
        if not self._client:
            raise RuntimeError("Sandbox not started")

        temp_dir = f"/tmp/{self._user_id}"

        list_code = textwrap.dedent(
            f"""
            import os, json
            _temp_dir = {temp_dir!r}
            if os.path.exists(_temp_dir):
                _files = [
                    f for f in os.listdir(_temp_dir)
                    if os.path.isfile(os.path.join(_temp_dir, f))
                ]
                print(json.dumps(_files))
            else:
                print(json.dumps([]))
        """
        ).strip()

        try:
            result = self._run(
                self._client.run_code(list_code, SandboxCodeOptions(language="python"))
            )
            if result.success:
                files: list[str] = json.loads(result.output.strip())
                return files
        except Exception as e:
            logger.error("Failed to list files: %s", e)

        return []

    # ------------------------------------------------------------------
    # Context manager / destructor
    # ------------------------------------------------------------------

    def __enter__(self) -> "PandasAISandboxAdapter":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def __del__(self) -> None:
        if getattr(self, "_started", False):
            self.stop()
