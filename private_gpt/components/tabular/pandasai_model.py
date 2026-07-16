import re
from datetime import datetime

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM
from pandasai.agent.state import AgentState  # ty:ignore[unresolved-import]
from pandasai.core.prompts import BasePrompt  # ty:ignore[unresolved-import]
from pandasai.llm import LLM as PandasAILLM  # ty:ignore[unresolved-import]


class PGPTPandasAILLM(PandasAILLM):  # type: ignore[misc]
    def __init__(
        self, llm: LLM, llm_alias: str | None = None, custom_prompt: str | None = None
    ) -> None:
        super().__init__(llm=llm)
        self._llm = llm
        self._llm_alias = llm_alias or "private_gpt"
        self._custom_prompt = custom_prompt or (
            "Generate SQL queries following these instructions: "
            f"Current date: {datetime.now().strftime('%Y-%m-%d')}. "
            "Use exact table/column names - this engine is case-sensitive. "
            "Before writing queries, infer how data is actually stored versus how users describe it. "
            "Handle data variations intelligently (e.g., user searches 'France' but table contains "
            "'French Republic' or 'FR'). Use LIKE patterns, wildcards, or multiple conditions "
            "to match similar values when exact matches may fail."
            "You must always return the type as the user expects, "
            "e.g., if the user expects a plot/chart, return a plot/chart, "
            "if the user expects a table, return a dataframe, etc."
            "The user does not want get partial responses, they want the final result, "
            "so do not return intermediate steps in result variable."
            "Import "
        )

    def call(self, instruction: BasePrompt, context: AgentState | None = None) -> str:
        prompt = f"{self._custom_prompt}\n" if self._custom_prompt else ""
        prompt += instruction.to_string()

        # Ensure to import all necessary libraries
        regex = r"import\s+pandas\s+as\s+pd"
        if re.search(regex, prompt):
            # replace pandas import by
            # matplotlib import, seaborn import, and pandas import
            prompt = re.sub(
                r"import\s+pandas\s+as\s+pd",
                "".join(
                    "import matplotlib.pyplot as plt\n"
                    "import seaborn as sns\n"
                    "import numpy as np\n"
                    "import pandas as pd",
                ),
                prompt,
            )

        result = self._llm.chat(
            messages=[
                ChatMessage(role=MessageRole.USER, content=prompt),
            ]
        )
        return str(result.message.content)

    @property
    def type(self) -> str:
        return self._llm_alias
