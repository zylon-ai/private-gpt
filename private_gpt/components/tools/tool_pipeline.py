from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.tools.processors.anthropic_tool_translation_processor import (
    AnthropicToolTranslationProcessor,
)
from private_gpt.components.tools.processors.bash_processor import BashProcessor
from private_gpt.components.tools.processors.code_execution_processor import (
    CodeExecutionProcessor,
)
from private_gpt.components.tools.processors.database_query_processor import (
    DatabaseQueryProcessor,
)
from private_gpt.components.tools.processors.present_files_processor import (
    PresentFilesProcessor,
)
from private_gpt.components.tools.processors.present_server_processor import (
    PresentServerProcessor,
)
from private_gpt.components.tools.processors.semantic_search_processor import (
    SemanticSearchProcessor,
)
from private_gpt.components.tools.processors.skill_management_processor import (
    SkillManagementProcessor,
)
from private_gpt.components.tools.processors.tabular_data_processor import (
    TabularDataProcessor,
)
from private_gpt.components.tools.processors.text_editor_processor import (
    TextEditorProcessor,
)
from private_gpt.components.tools.processors.web_fetch_processor import (
    WebFetchProcessor,
)
from private_gpt.components.tools.processors.web_search_processor import (
    WebSearchProcessor,
)


@singleton
class ToolPipeline:
    @inject
    def __init__(
        self,
        anthropic_tool_translation_processor: AnthropicToolTranslationProcessor,
        semantic_search_processor: SemanticSearchProcessor,
        tabular_data_processor: TabularDataProcessor,
        database_query_processor: DatabaseQueryProcessor,
        web_fetch_processor: WebFetchProcessor,
        web_search_processor: WebSearchProcessor,
        skill_management_processor: SkillManagementProcessor,
        code_execution_processor: CodeExecutionProcessor,
        bash_processor: BashProcessor,
        text_editor_processor: TextEditorProcessor,
        present_files_processor: PresentFilesProcessor,
        present_server_processor: PresentServerProcessor,
    ) -> None:
        self._processors = [
            anthropic_tool_translation_processor,
            semantic_search_processor,
            tabular_data_processor,
            database_query_processor,
            web_fetch_processor,
            web_search_processor,
            skill_management_processor,
            code_execution_processor,
            bash_processor,
            text_editor_processor,
            present_files_processor,
            present_server_processor,
        ]

    async def contextualize_internal_tools(
        self, request: ResolvedChatRequest
    ) -> ResolvedChatRequest:
        request_copy = request.model_copy(deep=True)
        await self._intercept_once(request_copy)
        return request_copy

    async def _intercept_once(self, request: ResolvedChatRequest) -> bool:
        for processor in self._processors:
            if await processor.intercept(request):
                return True
        return False
