from dataclasses import dataclass, field

from private_gpt.components.text_processing import (
    Action,
    IncrementalTextProcessor,
    ProbeResult,
    ProbeStatus,
    ProcessingContext,
)


@dataclass
class TokenRule:
    token: str
    action: Action
    replacement: str | None = None
    priority: int = 100
    name: str = "token"

    def probe(
        self, text: str, position: int, context: ProcessingContext
    ) -> ProbeResult:
        remaining = text[position:]
        if self.token.startswith(remaining) and len(remaining) < len(self.token):
            return ProbeResult.need_more()
        if not text.startswith(self.token, position):
            return ProbeResult.no_match()
        return ProbeResult(
            status=ProbeStatus.MATCH,
            consumed=len(self.token),
            action=self.action,
            replacement=self.replacement,
        )


def test_processor_passes_literal_text_without_rules() -> None:
    result = IncrementalTextProcessor([]).process("plain text")

    assert result.text == "plain text"
    assert result.pending == ""


def test_processor_supports_pass_drop_unwrap_and_replace() -> None:
    processor = IncrementalTextProcessor(
        [
            TokenRule("<pass>", Action.PASS),
            TokenRule("<drop>", Action.DROP),
            TokenRule("<unwrap>", Action.UNWRAP, "body"),
            TokenRule("<replace>", Action.REPLACE, "replacement"),
        ]
    )

    result = processor.process("<pass>|<drop>|<unwrap>|<replace>")

    assert result.text == "<pass>||body|replacement"


def test_need_more_holds_only_the_unresolved_suffix() -> None:
    processor = IncrementalTextProcessor(
        [TokenRule("<replace>", Action.REPLACE, "done")]
    )

    result = processor.process("safe <repl")

    assert result.text == "safe "
    assert result.pending == "<repl"
    assert result.consumed == len("safe ")


def test_finalization_can_resolve_rule_specific_partial_behavior() -> None:
    processor = IncrementalTextProcessor(
        [TokenRule("<replace>", Action.REPLACE, "done")]
    )

    result = processor.process("safe <repl", final=True)

    assert result.text == "safe "
    assert result.pending == "<repl"


def test_higher_priority_rule_wins_at_same_position() -> None:
    processor = IncrementalTextProcessor(
        [
            TokenRule("token", Action.REPLACE, "low", priority=10),
            TokenRule("token", Action.REPLACE, "high", priority=20),
        ]
    )

    assert processor.process("token").text == "high"


def test_feed_emits_only_new_safe_text() -> None:
    processor = IncrementalTextProcessor(
        [TokenRule("<replace>", Action.REPLACE, "done")]
    )

    first = processor.feed("before <rep")
    second = processor.feed("lace> after")

    assert first.text == "before "
    assert first.pending == "<rep"
    assert second.text == "done after"
    assert second.pending == ""


@dataclass
class StatefulRule:
    name: str = "stateful"
    priority: int = 100

    def probe(
        self, text: str, position: int, context: ProcessingContext
    ) -> ProbeResult:
        if not text.startswith("#", position):
            return ProbeResult.no_match()
        count = int(context.state.get("count", 0)) + 1
        return ProbeResult(
            status=ProbeStatus.MATCH,
            consumed=1,
            action=Action.REPLACE,
            replacement=str(count),
            state_updates={"count": count},
        )


def test_feed_reparses_from_clean_initial_rule_state() -> None:
    processor = IncrementalTextProcessor([StatefulRule()])

    assert processor.feed("#").text == "1"
    assert processor.feed("#").text == "2"
    assert processor.context.state == {"count": 2}


@dataclass
class TriggeredTokenRule(TokenRule):
    triggers: str = "<"
    probed: list[int] = field(default_factory=list)

    def probe(
        self, text: str, position: int, context: ProcessingContext
    ) -> ProbeResult:
        self.probed.append(position)
        return super().probe(text, position, context)


def test_rule_with_triggers_is_only_probed_at_trigger_characters() -> None:
    rule = TriggeredTokenRule("<replace>", Action.REPLACE, "done")
    processor = IncrementalTextProcessor([rule])

    result = processor.process("plain <replace> text < <rep")

    assert result.text == "plain done text < "
    assert result.pending == "<rep"
    assert rule.probed == [6, 21, 23]


def test_rule_without_triggers_is_probed_at_every_position() -> None:
    rule = TriggeredTokenRule("<replace>", Action.REPLACE, "done", triggers="")
    processor = IncrementalTextProcessor([rule])

    assert processor.process("a <b").text == "a <b"
    assert rule.probed == [0, 1, 2, 3]
