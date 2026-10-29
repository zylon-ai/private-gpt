"""Minimal repro of the tree-expansion token budget defect.

Run with the module on the path:
    python -m pytest tests/utils/test_token.py --noconftest
    python repro_token_budget.py
"""

from private_gpt.utils.token import calculate_maximum_token_expansion

# A 8192-token model, with the chat setting max_context_length = 2048.
CONTEXT_WINDOW = 8192
CONFIGURED_MAX_CONTEXT_LENGTH = 2048

CASES = [
    ("no caller limit, configured cap 2048", None, CONFIGURED_MAX_CONTEXT_LENGTH),
    (
        "caller limit 4096, configured cap 2048",
        4096,
        CONFIGURED_MAX_CONTEXT_LENGTH,
    ),
    (
        "caller limit 4096, no configured cap, model window 8192",
        4096,
        None,
    ),
]

print(f"model context_window = {CONTEXT_WINDOW}\n")
for label, token_limit, maximum_context_length in CASES:
    budget = calculate_maximum_token_expansion(
        token_limit=token_limit,
        context_window=CONTEXT_WINDOW,
        maximum_context_length=maximum_context_length,
    )
    limits = [n for n in (token_limit, maximum_context_length, CONTEXT_WINDOW) if n]
    expected = min(limits)
    verdict = "OK " if budget == expected else "BUG"
    print(
        f"[{verdict}] {label}\n"
        f"       token_limit={token_limit} maximum_context_length="
        f"{maximum_context_length} -> budget={budget} (tightest limit is {expected})"
    )
