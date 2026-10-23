from llama_index.core.base.llms.types import ChatMessage, MessageRole


def _is_left_tldr(msg: ChatMessage) -> bool:
    return msg.additional_kwargs.get("tldr") == "left"


def _is_right_tldr(msg: ChatMessage) -> bool:
    return msg.additional_kwargs.get("tldr") == "right"


def _trim_right_tldrs(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Per user-group: keep user + last contiguous right-TLDR run only.

    Everything else in the group (before and after the run) is dropped.
    Groups without a right TLDR are left unchanged.
    """
    groups: list[list[ChatMessage]] = []
    current: list[ChatMessage] = []
    for msg in messages:
        if msg.role == MessageRole.USER and current:
            groups.append(current)
            current = []
        current.append(msg)
    if current:
        groups.append(current)

    result: list[ChatMessage] = []
    for group in groups:
        if group[0].role != MessageRole.USER:
            result.extend(group)
            continue

        # Find the latest tldr right element
        # Get the max index of the right tldr in the group, if any
        latest_right_tldr_index = max(
            [i for i, msg in enumerate(group) if _is_right_tldr(msg)], default=-1
        )
        if latest_right_tldr_index == -1:
            result.extend(group)
        else:
            # Find the start of the contiguous right TLDR block
            right_tldr_block_start = latest_right_tldr_index
            for i in range(latest_right_tldr_index - 1, -1, -1):
                if not _is_right_tldr(group[i]):
                    break
                right_tldr_block_start = i

            result.append(group[0])  # Always keep the user message
            result.extend(
                group[right_tldr_block_start:]
            )  # Keep the following elements after this

    return result


def _trim_left_tldrs(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Find the last contiguous left-TLDR block and trim preceding content.

    - If the block starts with a user message: drop everything before it.
    - Otherwise: drop messages within the same user-group that precede the block,
      keeping content from the nearest user-starting left TLDR earlier in history.
    """
    if not messages:
        return []

    # Find the last user message
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break

    if last_user_idx == -1:
        return messages

    def _find_last_tldr_left(messages: list[ChatMessage]) -> int:
        """Find the last TLDR in a list of messages (for left side)."""
        max_tldr_idx: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            if _is_left_tldr(messages[i]):
                max_tldr_idx = i
                break

        if max_tldr_idx is None:
            return -1

        for i in range(max_tldr_idx, -1, -1):
            if _is_left_tldr(messages[i]):
                # Check if there's a next message and it's not a TLDR
                if 0 <= i - 1 < len(messages) and not _is_left_tldr(messages[i - 1]):
                    return i
        return -1

    def _find_first_tldr_right(messages: list[ChatMessage]) -> int:
        """Find the first TLDR in a list of messages (for right side).

        For right side, we keep everything from the first TLDR onwards,
        discarding only messages before the TLDR.
        """
        for i in range(len(messages)):
            if _is_left_tldr(messages[i]):
                return i
        return -1

    # Process messages before last user message (left side)
    before_user = messages[:last_user_idx]
    start_messages = []

    if before_user:
        tldr_idx = _find_last_tldr_left(before_user)
        start_messages = before_user[tldr_idx:] if tldr_idx != -1 else before_user

    # Process messages after last user message (right side)
    after_user = messages[last_user_idx + 1 :]
    end_messages = []

    if after_user:
        first_tldr_idx = _find_first_tldr_right(after_user)
        end_messages = (
            after_user[first_tldr_idx:] if first_tldr_idx != -1 else after_user
        )

    return [*start_messages, messages[last_user_idx], *end_messages]


def trim_to_last_tldr(messages: list[ChatMessage]) -> list[ChatMessage]:
    messages = _trim_left_tldrs(messages)
    return _trim_right_tldrs(messages)
