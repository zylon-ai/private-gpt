# mypy: ignore-errors
from __future__ import annotations

import io
import json
import logging
from typing import TYPE_CHECKING, Any

import boto3  # type: ignore
from llama_index.bridge.pydantic import Field
from llama_index.llms import (
    CompletionResponse,
    CustomLLM,
    LLMMetadata,
)
from llama_index.llms.base import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.llms.llama_utils import (
    completion_to_prompt as generic_completion_to_prompt,
)
from llama_index.llms.llama_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llama_index.callbacks import CallbackManager
    from llama_index.llms import (
        ChatMessage,
        ChatResponse,
        ChatResponseGen,
        CompletionResponseGen,
    )

logger = logging.getLogger(__name__)


class LineIterator:
    r"""A helper class for parsing the byte stream input from TGI container.

    The output of the model will be in the following format:
    ```
    b'data:{"token": {"text": " a"}}\n\n'
    b'data:{"token": {"text": " challenging"}}\n\n'
    b'data:{"token": {"text": " problem"
    b'}}'
    ...
    ```

    While usually each PayloadPart event from the event stream will contain a byte array
    with a full json, this is not guaranteed and some of the json objects may be split
    across PayloadPart events. For example:
    ```
    {'PayloadPart': {'Bytes': b'{"outputs": '}}
    {'PayloadPart': {'Bytes': b'[" problem"]}\n'}}
    ```


    This class accounts for this by concatenating bytes written via the 'write' function
    and then exposing a method which will return lines (ending with a '\n' character)
    within the buffer via the 'scan_lines' function. It maintains the position of the
    last read position to ensure that previous bytes are not exposed again. It will
    also save any pending lines that doe not end with a '\n' to make sure truncations
    are concatinated
    """

    def __init__(self, stream: Any) -> None:
        """Line iterator initializer."""
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self) -> Any:
        """Self iterator."""
        return self

    def __next__(self) -> Any:
        """Next element from iterator."""
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                logger.warning("Unknown event type=%s", chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])


class SagemakerLLM(CustomLLM):
    """Sagemaker Inference Endpoint models.

    To use, you must supply the endpoint name from your deployed
    Sagemaker model & the region where it is deployed.

    To authenticate, the AWS client uses the following methods to
    automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Sagemaker endpoint.
    See: https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html
    """

    endpoint_name: str = Field(description="")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_new_tokens: int = Field(description="The maximum number of tokens to generate.")
    context_window: int = Field(
        description="The maximum number of context tokens for the model."
    )
    messages_to_prompt: Any = Field(
        description="The function to convert messages to a prompt.", exclude=True
    )
    completion_to_prompt: Any = Field(
        description="The function to convert a completion to a prompt.", exclude=True
    )
    generate_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for generation."
    )
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for model initialization."
    )
    verbose: bool = Field(description="Whether to print verbose output.")

    _boto_client: Any = boto3.client(
        "sagemaker-runtime",
    )  # TODO make it an optional field

    def __init__(
        self,
        endpoint_name: str | None = "",
        temperature: float = 0.1,
        max_new_tokens: int = 512,  # to review defaults
        context_window: int = 2048,  # to review defaults
        messages_to_prompt: Any = None,
        completion_to_prompt: Any = None,
        callback_manager: CallbackManager | None = None,
        generate_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> None:
        """SagemakerLLM initializer."""
        model_kwargs = model_kwargs or {}
        model_kwargs.update({"n_ctx": context_window, "verbose": verbose})

        messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        completion_to_prompt = completion_to_prompt or generic_completion_to_prompt

        generate_kwargs = generate_kwargs or {}
        generate_kwargs.update(
            {"temperature": temperature, "max_tokens": max_new_tokens}
        )

        super().__init__(
            endpoint_name=endpoint_name,
            temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            callback_manager=callback_manager,
            generate_kwargs=generate_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
        )

    @property
    def inference_params(self):
        # TODO expose the rest of params
        return {
            "do_sample": True,
            "top_p": 0.7,
            "temperature": self.temperature,
            "top_k": 50,
            "max_new_tokens": self.max_new_tokens,
        }

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name="Sagemaker LLama 2",
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        self.generate_kwargs.update({"stream": False})

        is_formatted = kwargs.pop("formatted", False)
        if not is_formatted:
            prompt = self.completion_to_prompt(prompt)

        request_params = {
            "inputs": prompt,
            "stream": False,
            "parameters": self.inference_params,
        }

        resp = self._boto_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            Body=json.dumps(request_params),
            ContentType="application/json",
        )

        response_body = resp["Body"]
        response_str = response_body.read().decode("utf-8")
        response_dict = eval(response_str)

        return CompletionResponse(
            text=response_dict[0]["generated_text"][len(prompt) :], raw=resp
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        def get_stream():
            text = ""

            request_params = {
                "inputs": prompt,
                "stream": True,
                "parameters": self.inference_params,
            }
            resp = self._boto_client.invoke_endpoint_with_response_stream(
                EndpointName=self.endpoint_name,
                Body=json.dumps(request_params),
                ContentType="application/json",
            )

            event_stream = resp["Body"]
            start_json = b"{"
            stop_token = "<|endoftext|>"

            for line in LineIterator(event_stream):
                if line != b"" and start_json in line:
                    data = json.loads(line[line.find(start_json) :].decode("utf-8"))
                    if data["token"]["text"] != stop_token:
                        delta = data["token"]["text"]
                        text += delta
                        yield CompletionResponse(delta=delta, text=text, raw=data)

        return get_stream()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)
