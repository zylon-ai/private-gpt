"""Tests for TextInterceptor, ReasoningInterceptor, and ToolInterceptor."""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class ProcessingContext:
    token: str | None = None
    raw: str = ""
    previous_raw: str = ""
    text: str = ""
    additional_kwargs: dict[str, Any] = field(default_factory=dict)

    def model_copy_with_updates(self, **kwargs: Any) -> "ProcessingContext":
        copy = ProcessingContext(
            token=self.token,
            raw=self.raw,
            previous_raw=self.previous_raw,
            text=self.text,
            additional_kwargs=dict(self.additional_kwargs),
        )
        for k, v in kwargs.items():
            setattr(copy, k, v)
        return copy


@dataclass
class ProcessingConfig:
    pass


class BaseParserInterceptor:
    def process_streaming(
        self, context: ProcessingContext, config: ProcessingConfig, **kwargs: Any
    ) -> ProcessingContext:
        return context


@dataclass
class ParsedText:
    delta: str | None
    raw: str = ""

    class _Message:
        def __init__(self, content: str = "") -> None:
            self.content = content

    def __post_init__(self) -> None:
        self.message = self._Message(self.delta or "")


@dataclass
class ParsedReasoning:
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedTool:
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Interceptors under test
# ---------------------------------------------------------------------------


class TextInterceptor(BaseParserInterceptor):
    def __init__(self, text_parser=None) -> None:
        super().__init__()
        self._parser = text_parser
        self._previous_text = ""
        self._newline_buffer = ""

    def process_streaming(
        self, context: ProcessingContext, config: ProcessingConfig, **kwargs: Any
    ) -> ProcessingContext:
        if not self._parser or not context.token:
            return context

        parsed = self._parser.extract_text_content_streaming(
            previous_text=context.previous_raw,
            current_text=context.raw,
            delta_text=context.token,
        )
        delta = parsed.delta
        if delta is None:
            return context

        if not self._previous_text and not delta.strip():
            self._newline_buffer += delta
            return context

        self._newline_buffer = ""
        self._previous_text += delta
        return context.model_copy_with_updates(text=delta)


class ReasoningInterceptor(BaseParserInterceptor):
    def __init__(self, reasoning_parser=None):
        super().__init__()
        self._parser = reasoning_parser
        self._current_reasoning = ""
        self._newline_buffer = ""

    def process_streaming(
        self, context: ProcessingContext, config: ProcessingConfig, **kwargs: Any
    ) -> ProcessingContext:
        if not self._parser or not context.token:
            return context

        parsed = self._parser.extract_reasoning_content_streaming(
            previous_text=context.previous_raw,
            current_text=context.raw,
            delta_text=context.token,
        )
        if not parsed:
            return context

        reasoning_delta = parsed.additional_kwargs.get("thinking_delta")
        if reasoning_delta is None:
            self._current_reasoning = ""
            self._newline_buffer = ""
            return context

        if not self._current_reasoning and not reasoning_delta.strip():
            self._newline_buffer += reasoning_delta
            return context

        self._newline_buffer = ""
        self._current_reasoning += reasoning_delta
        return context.model_copy_with_updates(
            token="",
            additional_kwargs={
                **context.additional_kwargs,
                "thinking_delta": reasoning_delta,
            },
        )


class ToolInterceptor(BaseParserInterceptor):
    def __init__(self, tool_parser=None) -> None:
        super().__init__()
        self._parser = tool_parser
        self._has_tool_call = False

    def process_streaming(
        self, context: ProcessingContext, config: ProcessingConfig, **kwargs: Any
    ) -> ProcessingContext:
        if not self._parser or not context.token:
            return context

        parsed = self._parser.extract_tool_calls_streaming(
            previous_text=context.previous_raw,
            current_text=context.raw,
            delta_text=context.token,
        )
        if not parsed:
            return context

        tool_calls = parsed.additional_kwargs.get("tool_calls")
        if tool_calls:
            self._has_tool_call = True
            return context.model_copy_with_updates(
                token="",
                additional_kwargs={
                    **context.additional_kwargs,
                    "tool_calls": tool_calls,
                },
            )
        return context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_parser() -> MagicMock:
    parser = MagicMock()
    accumulated = [""]

    def extract(previous_text, current_text, delta_text):
        accumulated[0] += delta_text
        return ParsedText(delta=delta_text, raw=accumulated[0])

    parser.extract_text_content_streaming.side_effect = extract
    return parser


def _make_reasoning_parser() -> MagicMock:
    parser = MagicMock()

    def extract(previous_text, current_text, delta_text):
        return ParsedReasoning(additional_kwargs={"thinking_delta": delta_text})

    parser.extract_reasoning_content_streaming.side_effect = extract
    return parser


def _make_tool_parser(tool_token: str, tool_calls: list[Any]) -> MagicMock:
    parser = MagicMock()

    def extract(previous_text, current_text, delta_text):
        if delta_text == tool_token:
            return ParsedTool(additional_kwargs={"tool_calls": tool_calls})
        return None

    parser.extract_tool_calls_streaming.side_effect = extract
    return parser


def _run_text(tokens: list[str]) -> list[str]:
    interceptor = TextInterceptor(text_parser=_make_text_parser())
    config = ProcessingConfig()
    emitted, raw = [], ""
    for token in tokens:
        ctx = ProcessingContext(token=token, raw=raw + token, previous_raw=raw)
        result = interceptor.process_streaming(ctx, config)
        raw += token
        if result.text:
            emitted.append(result.text)
    return emitted


def _run_reasoning(tokens: list[str]) -> list[str]:
    interceptor = ReasoningInterceptor(reasoning_parser=_make_reasoning_parser())
    config = ProcessingConfig()
    emitted, raw = [], ""
    for token in tokens:
        ctx = ProcessingContext(token=token, raw=raw + token, previous_raw=raw)
        result = interceptor.process_streaming(ctx, config)
        raw += token
        if thinking := result.additional_kwargs.get("thinking_delta"):
            emitted.append(thinking)
    return emitted


def _run_tool_then_text(tool_token: str, text_tokens: list[str]) -> dict[str, Any]:
    tool_calls_fixture = [{"name": "get_weather", "arguments": {}}]
    tool_interceptor = ToolInterceptor(
        tool_parser=_make_tool_parser(tool_token, tool_calls_fixture)
    )
    text_interceptor = TextInterceptor(text_parser=_make_text_parser())
    config = ProcessingConfig()

    collected_tools, collected_text, raw = [], [], ""
    for token in [tool_token, *text_tokens]:
        ctx = ProcessingContext(token=token, raw=raw + token, previous_raw=raw)
        ctx = tool_interceptor.process_streaming(ctx, config)
        ctx = text_interceptor.process_streaming(ctx, config)
        raw += token
        if tc := ctx.additional_kwargs.get("tool_calls"):
            collected_tools.extend(tc)
        if ctx.text:
            collected_text.append(ctx.text)
    return {"tools": collected_tools, "text": collected_text}


def _run_tool_then_reasoning(
    tool_token: str, reasoning_tokens: list[str]
) -> dict[str, Any]:
    tool_calls_fixture = [{"name": "search", "arguments": {}}]
    tool_interceptor = ToolInterceptor(
        tool_parser=_make_tool_parser(tool_token, tool_calls_fixture)
    )
    reasoning_interceptor = ReasoningInterceptor(
        reasoning_parser=_make_reasoning_parser()
    )
    config = ProcessingConfig()

    collected_tools, collected_reasoning, raw = [], [], ""
    for token in [tool_token, *reasoning_tokens]:
        ctx = ProcessingContext(token=token, raw=raw + token, previous_raw=raw)
        ctx = tool_interceptor.process_streaming(ctx, config)
        ctx = reasoning_interceptor.process_streaming(ctx, config)
        raw += token
        if tc := ctx.additional_kwargs.get("tool_calls"):
            collected_tools.extend(tc)
        if thinking := ctx.additional_kwargs.get("thinking_delta"):
            collected_reasoning.append(thinking)
    return {"tools": collected_tools, "reasoning": collected_reasoning}


# ---------------------------------------------------------------------------
# TextInterceptor tests
# ---------------------------------------------------------------------------


class TestTextInterceptorNewlines:
    def test_leading_single_newline_stripped(self):
        assert _run_text(["\n", "hola"]) == ["hola"]

    def test_leading_multiple_newlines_stripped(self):
        assert _run_text(["\n", "\n", "hola"]) == ["hola"]

    def test_leading_newline_then_space_stripped(self):
        # Leading space is also whitespace — stripped along with \n
        assert "".join(_run_text(["\n", " ", "hola"])) == "hola"

    def test_leading_space_stripped(self):
        assert "".join(_run_text([" ", "hola"])) == "hola"

    def test_internal_newline_preserved(self):
        assert "".join(_run_text(["hola", "\n", "que"])) == "hola\nque"

    def test_internal_space_preserved(self):
        assert "".join(_run_text(["hola", " ", "que"])) == "hola que"

    def test_trailing_newline_emitted(self):
        assert "".join(_run_text(["hola", "\n"])) == "hola\n"

    def test_no_whitespace(self):
        assert _run_text(["hola", "que", "tal"]) == ["hola", "que", "tal"]

    def test_full_sentence_with_leading_newline(self):
        tokens = ["\n", "The", " ", "weather", " ", "is", " ", "5°C", "."]
        assert "".join(_run_text(tokens)) == "The weather is 5°C."

    def test_empty_token_passthrough(self):
        interceptor = TextInterceptor(text_parser=_make_text_parser())
        ctx = ProcessingContext(token="", raw="", previous_raw="")
        result = interceptor.process_streaming(ctx, ProcessingConfig())
        assert result.text == ""


# ---------------------------------------------------------------------------
# ReasoningInterceptor tests
# ---------------------------------------------------------------------------


class TestReasoningInterceptorNewlines:
    def test_leading_single_newline_stripped(self):
        assert _run_reasoning(["\n", "hola"]) == ["hola"]

    def test_leading_multiple_newlines_stripped(self):
        assert _run_reasoning(["\n", "\n", "hola"]) == ["hola"]

    def test_leading_newline_then_space_stripped(self):
        assert "".join(_run_reasoning(["\n", " ", "hola"])) == "hola"

    def test_leading_space_stripped(self):
        assert "".join(_run_reasoning([" ", "hola"])) == "hola"

    def test_internal_newline_preserved(self):
        assert "".join(_run_reasoning(["hola", "\n", "que"])) == "hola\nque"

    def test_internal_space_preserved(self):
        assert "".join(_run_reasoning(["hola", " ", "que"])) == "hola que"

    def test_trailing_newline_emitted(self):
        assert "".join(_run_reasoning(["hola", "\n"])) == "hola\n"

    def test_no_whitespace(self):
        assert _run_reasoning(["think", "step", "by"]) == ["think", "step", "by"]

    def test_reasoning_resets_on_mode_switch(self):
        interceptor = ReasoningInterceptor(reasoning_parser=MagicMock())

        def extract(previous_text, current_text, delta_text):
            if delta_text == "think":
                return ParsedReasoning(additional_kwargs={"thinking_delta": "think"})
            return ParsedReasoning(additional_kwargs={})

        interceptor._parser.extract_reasoning_content_streaming.side_effect = extract
        config = ProcessingConfig()

        ctx1 = ProcessingContext(token="think", raw="think", previous_raw="")
        interceptor.process_streaming(ctx1, config)
        assert interceptor._current_reasoning == "think"

        ctx2 = ProcessingContext(token="text", raw="thinktext", previous_raw="think")
        interceptor.process_streaming(ctx2, config)
        assert interceptor._current_reasoning == ""
        assert interceptor._newline_buffer == ""


# ---------------------------------------------------------------------------
# ToolInterceptor + Text pipeline tests
# ---------------------------------------------------------------------------


class TestToolThenTextPipeline:
    def test_tool_detected_then_text(self):
        result = _run_tool_then_text(
            tool_token='{"tool": "get_weather"}',
            text_tokens=["The", " ", "weather", " ", "is", " ", "5°C"],
        )
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert "".join(result["text"]) == "The weather is 5°C"

    def test_tool_detected_then_text_with_leading_newline(self):
        result = _run_tool_then_text(
            tool_token='{"tool": "get_weather"}',
            text_tokens=["\n", "The", " ", "weather"],
        )
        assert len(result["tools"]) == 1
        assert "".join(result["text"]) == "The weather"

    def test_tool_detected_then_text_with_leading_space(self):
        result = _run_tool_then_text(
            tool_token='{"tool": "get_weather"}',
            text_tokens=[" ", "The", " ", "weather"],
        )
        assert len(result["tools"]) == 1
        assert "".join(result["text"]) == "The weather"

    def test_tool_suppresses_token(self):
        result = _run_tool_then_text(
            tool_token='{"tool": "get_weather"}',
            text_tokens=["hello"],
        )
        assert '{"tool"' not in "".join(result["text"])

    def test_no_tool_only_text(self):
        tool_interceptor = ToolInterceptor(tool_parser=None)
        text_interceptor = TextInterceptor(text_parser=_make_text_parser())
        config = ProcessingConfig()

        collected_text, raw = [], ""
        for token in ["hello", " ", "world"]:
            ctx = ProcessingContext(token=token, raw=raw + token, previous_raw=raw)
            ctx = tool_interceptor.process_streaming(ctx, config)
            ctx = text_interceptor.process_streaming(ctx, config)
            raw += token
            if ctx.text:
                collected_text.append(ctx.text)

        assert "".join(collected_text) == "hello world"


# ---------------------------------------------------------------------------
# ToolInterceptor + Reasoning pipeline tests
# ---------------------------------------------------------------------------


class TestToolThenReasoningPipeline:
    def test_tool_then_reasoning(self):
        result = _run_tool_then_reasoning(
            tool_token='{"tool": "search"}',
            reasoning_tokens=["think", " ", "carefully"],
        )
        assert len(result["tools"]) == 1
        assert "".join(result["reasoning"]) == "think carefully"

    def test_tool_then_reasoning_with_leading_newline(self):
        result = _run_tool_then_reasoning(
            tool_token='{"tool": "search"}',
            reasoning_tokens=["\n", "think"],
        )
        assert len(result["tools"]) == 1
        assert "".join(result["reasoning"]) == "think"

    def test_tool_then_reasoning_with_leading_space(self):
        result = _run_tool_then_reasoning(
            tool_token='{"tool": "search"}',
            reasoning_tokens=[" ", "think"],
        )
        assert len(result["tools"]) == 1
        assert "".join(result["reasoning"]) == "think"
