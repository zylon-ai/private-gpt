def calculate_maximum_token_expansion(
    token_limit: int | None,
    context_window: int,
    maximum_context_length: int | None = None,
) -> int:
    if token_limit is None:
        return context_window
    elif maximum_context_length is not None and token_limit > maximum_context_length:
        return maximum_context_length
    else:
        return token_limit
