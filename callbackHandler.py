from typing import Any, Union
from asyncio.queues import Queue
from langchain.callbacks.base import AsyncCallbackHandler
from typing import Any, Dict, List, Optional
from langchain.schema import LLMResult
import json
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler


DEFAULT_ANSWER_PREFIX_TOKENS = ["\n", "AI", ":"]

class CustomAsyncCallBackHandler(AsyncCallbackHandler):
    queue: Queue
    
    
    def __init__(self,queue:Queue, answer_prefix_tokens: Optional[List[str]] = None) -> None:
        super().__init__()
        if answer_prefix_tokens is None:
            answer_prefix_tokens = DEFAULT_ANSWER_PREFIX_TOKENS
        self.answer_prefix_tokens = answer_prefix_tokens
        self.last_tokens = [""] * len(answer_prefix_tokens)
        self.answer_reached = False
        self.queue=queue


    async def put_message(self,json_str):
        await self.queue.put(json.dumps(json_str))
    #         # 等待所有消息发送完成
        await self.queue.join()

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.answer_reached = False

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
         # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.last_tokens.append(token)
        if len(self.last_tokens) > len(self.answer_prefix_tokens):
            self.last_tokens.pop(0)

        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.last_tokens == self.answer_prefix_tokens:
            self.answer_reached = True
            # Do not print the last token in answer_prefix_tokens,
            # as it's not part of the answer yet
            return

        # ... if yes, then print tokens from now on
        if self.answer_reached:
            response = {"stat": "SUCCESS", "message": str(token)}
            await self.put_message(response)

        

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        if self.answer_reached:
            response = {"stat": "SUCCESS", "message": "[DONE]"}
            #sys.stdout.write(str(response))
            #sys.stdout.flush()
            print(str(response))
            await self.put_message(response)

    async def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        
        
# llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", verbose=False,streaming=True,callbacks=[CustomAsyncCallBackHandler(message_queue)])