def calculate_maximum_token_expansion(
    token_limit: int | None,
    context_window: int,
    maximum_context_length: int | None = None,
) -> int:
    """Budget for tree expansion: the tightest of every applicable limit.

    ``context_window`` bounds the model, ``maximum_context_length`` the
    configured chat context and ``token_limit`` the caller (a retrieve request
    or tool invocation). None of them may be raised, so the budget is their
    minimum; a None limit simply does not constrain.
    """
    limits = [context_window]
    if token_limit is not None:
        limits.append(token_limit)
    if maximum_context_length is not None:
        limits.append(maximum_context_length)

    return min(limits)
