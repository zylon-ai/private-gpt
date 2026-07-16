import importlib
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeAlias, assert_never, cast

import huggingface_hub  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
from huggingface_hub import (  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
    HfApi,
    hf_hub_download,
)

from private_gpt.components.llm.tokenizers.tokenizer_base import (
    AudioLike,
    ImageLike,
    TextLike,
    TokenizedInput,
    TokenizerBase,
)
from private_gpt.utils.dependencies import format_missing_dependency_message

ChatMessageType: TypeAlias = Any
Tokenized: TypeAlias = Any
PublicMistralTokenizer: TypeAlias = Any


@lru_cache
def _load_mistral_module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "Mistral tokenizer",
                extras="llm-mistral",
            )
        ) from e


_messages_module = _load_mistral_module("mistral_common.protocol.instruct.messages")
AssistantMessage = _messages_module.AssistantMessage
SystemMessage = _messages_module.SystemMessage
ToolMessage = _messages_module.ToolMessage
UserMessage = _messages_module.UserMessage

logger = logging.getLogger(__name__)


def is_list_of(
    value: object,
    typ: type[Any],
    *,
    check: Literal["first", "all"] = "first",
) -> bool:
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)

    assert_never(check)


def maybe_serialize_tool_calls(request: Any) -> None:
    """Workaround for Pydantic iterator bug with tool_calls.

    SEE: https://github.com/vllm-project/vllm/pull/9951
    Credits: @gcalmettes

    There is a bug in Pydantic where attributes declared as iterables are replaced
    by ValidatorIterator instances. This affects tool_calls in assistant messages.

    Official Pydantic Issues:
    - https://github.com/pydantic/pydantic/issues/9467
    - https://github.com/pydantic/pydantic/issues/9541

    TODO: Remove when Pydantic v2.11+ is released
    """
    new_messages: list[Any] = []

    for message_obj in request.messages:
        message = message_obj.model_dump()
        if message.get("role") == "assistant":
            tool_calls_validator = message.get("tool_calls") or []
            tool_calls_iter = tool_calls_validator.__iter__()
            validated_tool_calls = []
            while tool_calls_iter:
                try:
                    tool_call = next(tool_calls_iter)
                    validated_tool_calls.append(tool_call)
                except StopIteration:
                    break

            if validated_tool_calls:
                message["tool_calls"] = validated_tool_calls
                new_messages.append(AssistantMessage(**message))  # type: ignore[arg-type]
            else:
                new_messages.append(message_obj)
        else:
            new_messages.append(message_obj)

    request.messages = new_messages  # type: ignore[assignment]


def truncate_tool_call_ids(request: Any) -> None:
    """Truncates tool call IDs to meet Mistral's 9-character limit."""
    new_messages: list[Any] = []

    for message_obj in request.messages:
        message = message_obj.model_dump()
        modified = False

        if message.get("role") == "assistant":
            tool_calls = message.get("tool_calls") or []
            for tool_call in tool_calls:
                if len(tool_call["id"]) > 9:
                    logger.warning(
                        "Truncating tool call ID: %s to %s",
                        tool_call["id"],
                        tool_call["id"][-9:],
                    )
                    tool_call["id"] = tool_call["id"][-9:]
                    modified = True

            if modified:
                message["tool_calls"] = tool_calls
                new_messages.append(AssistantMessage(**message))  # type: ignore[arg-type]
            else:
                new_messages.append(message_obj)

        elif message.get("role") in {"tool_results", "tool"}:
            if "tool_call_id" in message:
                tool_call_id = message["tool_call_id"]

                if len(tool_call_id) > 9:
                    logger.warning(
                        "Truncating tool_call_id: %s to %s",
                        tool_call_id,
                        tool_call_id[-9:],
                    )
                    tool_call_id = tool_call_id[-9:]
                    message["tool_call_id"] = tool_call_id
                    modified = True

            if modified:
                new_messages.append(ToolMessage(**message))  # type: ignore[arg-type]
            else:
                new_messages.append(message_obj)
        else:
            new_messages.append(message_obj)

    request.messages = new_messages  # type: ignore[assignment]


def list_local_repo_files(repo_id: str, revision: str | None) -> list[str]:
    """List files in a locally cached HuggingFace repository."""
    repo_cache = os.path.join(
        huggingface_hub.constants.HF_HUB_CACHE,
        huggingface_hub.constants.REPO_ID_SEPARATOR.join(
            ["models", *repo_id.split("/")]
        ),
    )

    if revision is None:
        revision_file = os.path.join(repo_cache, "refs", "main")
        if os.path.isfile(revision_file):
            with open(revision_file) as file:
                revision = file.read()

    if revision:
        revision_dir = os.path.join(repo_cache, "snapshots", revision)
        if os.path.isdir(revision_dir):
            return os.listdir(revision_dir)

    return []


def find_tokenizer_file(files: list[str]) -> str:
    """Find the Mistral tokenizer file from a list of repository files."""
    file_pattern = re.compile(
        r"^tokenizer\.model\.v.*$|^tekken\.json$|^tokenizer\.mm\.model\.v.*$"
    )

    matched_files = [file for file in files if file_pattern.match(file)]
    if len(matched_files) > 1:
        raise OSError(
            f"Found {len(matched_files)} files matching the "
            f"pattern: {file_pattern}. Make sure only one Mistral "
            f"tokenizer is present in {files}."
        )
    elif len(matched_files) == 0:
        raise OSError(
            f"Found {len(matched_files)} files matching the "
            f"pattern: {file_pattern}. Make sure that a Mistral "
            f"tokenizer is present in {files}."
        )

    return matched_files[0]


def _prepare_apply_chat_template_tools_and_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    continue_final_message: bool = False,
    add_generation_prompt: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Prepare messages and tools for Mistral's chat template format.

    Handles validation and formatting of messages and tools to ensure
    compatibility with Mistral's requirements.
    """
    tool_calls_module = _load_mistral_module(
        "mistral_common.protocol.instruct.tool_calls"
    )
    Function = tool_calls_module.Function
    Tool = tool_calls_module.Tool

    if add_generation_prompt and continue_final_message:
        raise ValueError(
            "Cannot set both `add_generation_prompt` and "
            "`continue_final_message` to True."
        )

    last_message = messages[-1]

    if add_generation_prompt and last_message["role"] == "assistant":
        raise ValueError(
            "Cannot set `add_generation_prompt` to True when "
            "the last message is from the assistant. Consider "
            "using `continue_final_message` instead."
        )

    if continue_final_message and last_message["role"] != "assistant":
        raise ValueError(
            "Cannot set `continue_final_message` to True when "
            "the last message is not from the assistant."
        )

    # Mistral-common requires AssistantMessage content to be string
    # https://github.com/mistralai/mistral-common/blob/f4a06998b75ed78bbf5aaf569590b772ea26c9f6/src/mistral_common/protocol/instruct/messages.py#L80
    for message in messages:
        # Remove reasoning as unsupported by Mistral
        _ = message.pop("thinking", None)

        # Convert assistant message content to string if needed
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, list):
                content = "\n".join(chunk.get("text") or "" for chunk in content)
                message["content"] = content

    # Mistral requires "parameters" and "description" to be present even if empty
    if tools:
        for function in [
            tool["function"] for tool in tools if tool["type"] == "function"
        ]:
            if function.get("parameters") is None:
                function["parameters"] = {}
            if function.get("description") is None:
                function["description"] = ""

        # Filter unsupported arguments for mistral-common compatibility
        tools_fields = set(Tool.model_fields.keys())
        function_fields = set(Function.model_fields.keys())

        for tool in tools:
            tool_keys = list(tool.keys())
            for tool_key in tool_keys:
                if tool_key not in tools_fields:
                    tool.pop(tool_key)
                    logger.warning(
                        "'%s' is not supported by mistral-common for tools. "
                        "It has been removed from the tool definition.",
                        tool_key,
                    )

                if tool["type"] == "function":
                    function_keys = list(tool["function"].keys())
                    for function_key in function_keys:
                        if function_key not in function_fields:
                            tool["function"].pop(function_key)
                            logger.warning(
                                "'%s' is not supported by mistral-common "
                                "for function tools. It has been removed from the "
                                "function definition.",
                                function_key,
                            )
                else:
                    raise ValueError("mistral-common only supports function tools.")

    return messages, tools


def _tekken_token_to_id(tokenizer: Any, token: str | bytes) -> int:
    """Convert a Tekken token to its ID, with fallback to UNK."""
    Tekkenizer = _load_mistral_module(
        "mistral_common.tokens.tokenizers.tekken"
    ).Tekkenizer

    assert isinstance(tokenizer, Tekkenizer), type(tokenizer)

    token_bytes = token.encode("utf-8") if not isinstance(token, bytes) else token
    shift = tokenizer.num_special_tokens

    try:
        return cast(int, shift + tokenizer._tekken_token2id_nospecial[token_bytes])
    except KeyError:
        token_str = token_bytes.decode("utf-8")
        if token_str in tokenizer._special_tokens_reverse_vocab:
            return cast(int, tokenizer._special_tokens_reverse_vocab[token_str])

        logger.warning(
            "Failed to convert token %s to id, replacing with <unk>",
            token_bytes,
        )
        return cast(int, tokenizer.unk_id)


class MistralTokenizer(TokenizerBase):
    """Production-ready Mistral tokenizer implementation.

    Supports both Tekken and SentencePiece tokenizers with full compatibility
    for vLLM and structured output backends.
    """

    def __init__(
        self,
        tokenizer: Any,
    ) -> None:
        ValidationMode = _load_mistral_module(
            "mistral_common.protocol.instruct.validator"
        ).ValidationMode
        SentencePieceTokenizer = _load_mistral_module(
            "mistral_common.tokens.tokenizers.sentencepiece"
        ).SentencePieceTokenizer
        Tekkenizer = _load_mistral_module(
            "mistral_common.tokens.tokenizers.tekken"
        ).Tekkenizer

        self.mistral = tokenizer
        self.instruct = tokenizer.instruct_tokenizer
        self.tokenizer = self.instruct.tokenizer

        # Ensure test mode for proper validation
        mode = tokenizer._chat_completion_request_validator._mode
        if mode != ValidationMode.test:
            raise ValueError(
                "Mistral tokenizer must be in test mode. Set "
                "`mode=ValidationMode.test` when creating the tokenizer."
            )

        _mistral_version_str = str(self.tokenizer.version.value)
        self.version: int = int(_mistral_version_str.split("v")[-1])

        self.is_tekken = isinstance(self.tokenizer, Tekkenizer)
        self.is_spm = isinstance(self.tokenizer, SentencePieceTokenizer)

        if not (self.is_tekken or self.is_spm):
            raise TypeError(f"Unsupported tokenizer: {type(self.tokenizer)}")

        # Build vocabulary dict (reverse order to keep lowest token id)
        self._vocab = self.tokenizer.vocab()
        self._max_token_id = self.vocab_size - 1

        self._vocab_dict = {
            self.convert_ids_to_tokens([i], skip_special_tokens=False)[0]: i
            for i in range(self.vocab_size - 1, -1, -1)
        }
        self._vocab_dict = dict(sorted(self._vocab_dict.items(), key=lambda x: x[1]))

        # Cache special tokens for performance
        self._special_token_ids = self._get_special_token_ids()
        self._special_token_ids_set = set(self._special_token_ids)
        self._special_tokens = self._get_special_tokens(self._special_token_ids)
        self._special_tokens_set = set(self._special_tokens)

    def _get_special_token_ids(self) -> list[int]:
        """Extract all special token IDs from the tokenizer."""
        SentencePieceTokenizer = _load_mistral_module(
            "mistral_common.tokens.tokenizers.sentencepiece"
        ).SentencePieceTokenizer
        Tekkenizer = _load_mistral_module(
            "mistral_common.tokens.tokenizers.tekken"
        ).Tekkenizer

        if self.is_tekken:
            assert isinstance(self.tokenizer, Tekkenizer), type(self.tokenizer)
            special_ids = {t["rank"] for t in self.tokenizer._all_special_tokens}
        elif self.is_spm:
            assert isinstance(self.tokenizer, SentencePieceTokenizer), type(
                self.tokenizer
            )
            special_ids = self.tokenizer._control_tokens
        else:
            raise ValueError(f"Unknown tokenizer type: {type(self.tokenizer)}")

        return sorted(special_ids)

    def _get_special_tokens(self, all_special_ids: list[int]) -> list[str]:
        """Decode special token IDs to their string representations."""
        SpecialTokenPolicy = _load_mistral_module(
            "mistral_common.tokens.tokenizers.base"
        ).SpecialTokenPolicy

        return [
            self.tokenizer.decode([i], special_token_policy=SpecialTokenPolicy.KEEP)
            for i in all_special_ids
        ]

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> "MistralTokenizer":
        """Load tokenizer from local path or HuggingFace Hub."""
        if isinstance(model_id, str) and os.path.exists(model_id):
            # This is a local filesystem path, convert string to Path
            model_id = Path(model_id)

        if "trust_remote_code" in kwargs:
            logger.warning(
                "The 'trust_remote_code' argument is not applicable for Mistral tokenizers and will be ignored."
            )
            kwargs.pop("trust_remote_code")

        if isinstance(model_id, str) or not model_id.exists():
            model_id_str = str(model_id)
            assert not local_files_only, (
                "local_files_only=True but model_id is not a local path"
            )
            assert len(model_id_str.split("/")) == 2, (
                f"You have either provided a non-existent path: "
                f"{model_id} or an invalid HF Hub repo id."
            )
            tokenizer_file = cls._download_mistral_tokenizer_from_hf(
                model_id_str, revision, **kwargs
            )
        elif model_id.is_dir():
            tokenizer_file_name = find_tokenizer_file(os.listdir(model_id))
            tokenizer_file = str(Path(model_id) / tokenizer_file_name)
        else:
            assert Path(model_id).is_file(), f"Invalid path: {model_id}"
            tokenizer_file = str(model_id)

        ValidationMode = _load_mistral_module(
            "mistral_common.protocol.instruct.validator"
        ).ValidationMode
        PublicMistralTokenizer = _load_mistral_module(
            "mistral_common.tokens.tokenizers.mistral"
        ).MistralTokenizer

        mistral_tokenizer = PublicMistralTokenizer.from_file(
            tokenizer_file,
            mode=ValidationMode.test,
        )
        return cls(mistral_tokenizer)

    @classmethod
    def is_available(cls, model_id: str | Path | None, **kwargs: Any) -> bool:
        return model_id is not None

    @staticmethod
    def _download_mistral_tokenizer_from_hf(
        tokenizer_name: str,
        revision: str | None,
        **kwargs: Any,
    ) -> str:
        """Download Mistral tokenizer from HuggingFace Hub."""
        try:
            hf_api = HfApi()
            files = hf_api.list_repo_files(repo_id=tokenizer_name, revision=revision)
        except ConnectionError as exc:
            files = list_local_repo_files(repo_id=tokenizer_name, revision=revision)
            if len(files) == 0:
                raise exc

        filename = find_tokenizer_file(files)
        tokenizer_file: str = hf_hub_download(
            tokenizer_name,
            filename=filename,
            revision=revision,
            **kwargs,
        )
        return tokenizer_file

    @property
    def all_special_tokens(self) -> list[str]:
        return self._special_tokens

    @property
    def all_special_ids(self) -> list[int]:
        return self._special_token_ids

    @property
    def bos_token_id(self) -> int:
        return cast(int, self.tokenizer.bos_id)

    @property
    def eos_token_id(self) -> int:
        return cast(int, self.tokenizer.eos_id)

    @property
    def pad_token_id(self) -> int:
        return cast(int, self.tokenizer.pad_id)

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def max_token_id(self) -> int:
        return self._max_token_id

    @property
    def is_multimodal(self) -> bool:
        return (
            self.instruct.image_encoder is not None
            or self.instruct.audio_encoder is not None
        )

    def _is_special_token_id(self, token_id: int) -> bool:
        return token_id in self._special_token_ids_set

    def __hash__(self) -> int:
        return hash(id(self))

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        texts: TextLike | None = None,
        images: ImageLike | None = None,
        audios: AudioLike | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> TokenizedInput:
        """Encode text to token IDs."""
        text_input_ids: list[int] = []
        mm_input_ids: list[int] = []

        if texts:
            text_input_ids = self.calculate_text_input_ids(texts=texts)

        if images or audios:
            mm_input_ids = self.calculate_mm_input_ids(
                texts=texts,
                images=images,
                audios=audios,
            )

        return TokenizedInput(
            input_ids=text_input_ids + mm_input_ids,
        )

    def calculate_text_input_ids(
        self,
        texts: TextLike | None = None,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Estimate tokens for text using tokenizer."""
        if not texts:
            return []

        input_ids_: list[list[int]] = []
        for p in texts:
            each_input_ids = self.encode(str(p), add_special_tokens)
            if truncation and max_length:
                each_input_ids = each_input_ids[:max_length]
            input_ids_.append(each_input_ids)

        return [id for sublist in input_ids_ for id in sublist]

    def calculate_mm_input_ids(
        self,
        texts: TextLike | None = None,
        images: ImageLike | None = None,
        audios: AudioLike | None = None,
    ) -> list[int]:
        """Estimate tokens for images and audio using processor."""
        # TODO: Not supported yet as Mistral's multimodal tokenization is not finalized.
        return []

    def get_vocab(self) -> dict[str, int]:
        """Get vocabulary as dict (lossy for tokens with same string representation)."""
        return self._vocab_dict

    def get_added_vocab(self) -> dict[str, int]:
        """Mistral tokenizers have no added vocabulary."""
        return {}

    def encode(
        self,
        text: str,
        add_special_tokens: bool | None = None,
    ) -> list[int]:
        """Encode text to token IDs.

        Should only be used for prompt completion, not chat completion.
        For chat completion, use `apply_chat_template`.
        """
        if add_special_tokens is None:
            add_special_tokens = True
        return cast(list[int], self.tokenizer.encode(text, bos=False, eos=False))

    def apply_chat_template(
        self,
        conversation: list[dict[str, str | list[dict[str, str]]]],
        tools: list[dict[str, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> list[int] | str:
        """Apply chat template and optionally tokenize."""
        messages, tools = _prepare_apply_chat_template_tools_and_messages(
            conversation,
            tools,
            continue_final_message,
            add_generation_prompt,
        )

        ChatCompletionRequest = _load_mistral_module(
            "mistral_common.protocol.instruct.request"
        ).ChatCompletionRequest
        Tool = _load_mistral_module("mistral_common.protocol.instruct.tool_calls").Tool

        request: Any = ChatCompletionRequest(
            messages=messages,  # type: ignore[arg-type]
            tools=[Tool(**tool) for tool in tools] if tools else None,
        )

        # Apply pydantic workaround
        maybe_serialize_tool_calls(request)
        truncate_tool_call_ids(request)

        encoded = self.mistral.encode_chat_completion(request)

        if tokenize:
            tokens: list[int] = encoded.tokens
            return tokens
        else:
            result = cast(str | None, encoded.text)
            if not result:
                raise ValueError("Empty response from Mistral tokenizer")
            return result

    def decode(
        self,
        ids: list[int] | int,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        if isinstance(ids, int):
            ids = [ids]

        if not skip_special_tokens:
            SpecialTokenPolicy = _load_mistral_module(
                "mistral_common.tokens.tokenizers.base"
            ).SpecialTokenPolicy

            return cast(str, self.tokenizer.decode(ids, SpecialTokenPolicy.KEEP))

        return cast(str, self.tokenizer.decode(ids))

    def batch_decode(
        self,
        ids: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode multiple sequences of token IDs."""
        return [self.decode(seq, skip_special_tokens) for seq in ids]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Convert token strings back to text.

        Handles both Tekken and SentencePiece tokenizers with proper
        special token handling.
        """
        tokenizer_base_module = _load_mistral_module(
            "mistral_common.tokens.tokenizers.base"
        )
        SpecialTokenPolicy = tokenizer_base_module.SpecialTokenPolicy
        SpecialTokens = tokenizer_base_module.SpecialTokens
        SentencePieceTokenizer = _load_mistral_module(
            "mistral_common.tokens.tokenizers.sentencepiece"
        ).SentencePieceTokenizer
        Tekkenizer = _load_mistral_module(
            "mistral_common.tokens.tokenizers.tekken"
        ).Tekkenizer

        to_decode_special_tokens = {SpecialTokens.tool_calls}

        if self.is_tekken:
            assert isinstance(self.tokenizer, Tekkenizer), type(self.tokenizer)

            tokens = [
                t
                for t in tokens
                if (t in to_decode_special_tokens or t not in self._special_tokens_set)
            ]

            if any(isinstance(t, bytes) for t in tokens):
                # Need to encode and decode all tokens again for bytes
                ids = [_tekken_token_to_id(self.tokenizer, t) for t in tokens]
                decoded = cast(str, self.tokenizer.decode(ids, SpecialTokenPolicy.KEEP))
            else:
                decoded = "".join(tokens)

        else:
            assert isinstance(self.tokenizer, SentencePieceTokenizer), type(
                self.tokenizer
            )

            regular_tokens: list[str] = []
            decoded_list: list[str] = []

            for token in tokens:
                if token in to_decode_special_tokens:
                    if regular_tokens:
                        # Convert token strings back to IDs for decoding
                        regular_token_ids = [
                            self.get_vocab().get(t, self.tokenizer.unk_id)
                            for t in regular_tokens
                        ]
                        decoded_list.append(
                            cast(
                                str,
                                self.tokenizer.decode(
                                    regular_token_ids,
                                    SpecialTokenPolicy.IGNORE,
                                ),
                            )
                        )
                        regular_tokens = []
                    decoded_list.append(token)
                else:
                    regular_tokens.append(token)

            if regular_tokens:
                # Convert token strings back to IDs for decoding
                regular_token_ids = [
                    self.get_vocab().get(t, self.tokenizer.unk_id)
                    for t in regular_tokens
                ]
                decoded_list.append(
                    cast(
                        str,
                        self.tokenizer.decode(
                            regular_token_ids, SpecialTokenPolicy.IGNORE
                        ),
                    )
                )

            decoded = "".join(decoded_list)

        return decoded

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Convert token IDs to token strings.

        Handles incomplete UTF-8 sequences for Tekken tokenizer.
        """
        tokenizer_base_module = _load_mistral_module(
            "mistral_common.tokens.tokenizers.base"
        )
        SpecialTokenPolicy = tokenizer_base_module.SpecialTokenPolicy
        SpecialTokens = tokenizer_base_module.SpecialTokens
        InstructTokenizerV13 = _load_mistral_module(
            "mistral_common.tokens.tokenizers.instruct"
        ).InstructTokenizerV13

        if not skip_special_tokens:
            return [self.tokenizer.id_to_piece(token_id) for token_id in ids]

        # Keep certain special tokens like tool_calls
        non_skip_special_tokens_ids = {
            self.tokenizer.get_control_token(SpecialTokens.tool_calls),
        }

        if isinstance(self.instruct, InstructTokenizerV13):
            if self.instruct.BEGIN_THINK:
                non_skip_special_tokens_ids.add(self.instruct.BEGIN_THINK)
            if self.instruct.END_THINK:
                non_skip_special_tokens_ids.add(self.instruct.END_THINK)

        ids_kept = [
            i
            for i in ids
            if i in non_skip_special_tokens_ids or not self._is_special_token_id(i)
        ]

        tokens = [self.tokenizer.id_to_piece(token_id) for token_id in ids_kept]

        # Handle incomplete UTF-8 for Tekken
        if any("�" in t for t in tokens) and self.is_tekken:
            # Token contains replacement character, use bytes instead
            # See: https://github.com/vllm-project/vllm/pull/8640
            #      https://github.com/vllm-project/vllm/pull/9625
            Tekkenizer = _load_mistral_module(
                "mistral_common.tokens.tokenizers.tekken"
            ).Tekkenizer

            assert isinstance(self.tokenizer, Tekkenizer)
            tokens_final: list[str] = []
            for token_id in ids_kept:
                if token_id not in self._special_token_ids_set:
                    byte_token = self.tokenizer.id_to_byte_piece(
                        token_id, SpecialTokenPolicy.KEEP
                    )
                    # Convert bytes to str for type consistency
                    if isinstance(byte_token, bytes):
                        tokens_final.append(
                            byte_token.decode("utf-8", errors="replace")
                        )
                    else:
                        tokens_final.append(byte_token)
                else:
                    tokens_final.append(
                        self.tokenizer.decode([token_id], SpecialTokenPolicy.KEEP)
                    )
            return tokens_final

        return tokens
