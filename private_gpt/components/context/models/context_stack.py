from llama_index.core.base.llms.types import TextBlock
from pydantic import BaseModel, ConfigDict, Field

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.context.models.context_layer import (
    AnyContextLayer,
    ContentBundlesLayer,
    DocumentLayer,
    ToolDefinitionsLayer,
)
from private_gpt.components.context.models.layer_type import LayerType
from private_gpt.components.engines.citations.types import Document
from private_gpt.components.sandbox.content_bundle import ContentBundle


class ContextStack(BaseModel):
    """Ordered, immutable assembly of typed context layers for one request.

    Interceptors grow the stack each iteration by calling ``append_layer()``
    or ``append_layers()``, which return new instances (the stack is frozen).

    The engine reads:
    - ``stack.to_system_prompt()``  → system message text (prompt layers only)
    - ``stack.all_tools()``         → flat deduplicated list of ToolSpec
    """

    layers: list[AnyContextLayer] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def to_system_prompt(self) -> list[TextBlock]:
        """Render prompt layers by priority (then insertion order)."""
        ordered_layers = sorted(
            enumerate(self.layers),
            key=lambda item: (item[1].priority, item[0]),
        )
        chunks = [
            rendered
            for _, layer in ordered_layers
            for rendered in [layer.render()]
            if rendered.strip()
        ]
        return [TextBlock(text=chunk) for chunk in chunks]

    def all_tools(self) -> list[ToolSpec]:
        """Return deduplicated ToolSpec list from all TOOL_DEFINITIONS layers."""
        seen: set[str | None] = set()
        result: list[ToolSpec] = []
        for layer in self.layers:
            if not isinstance(layer, ToolDefinitionsLayer):
                continue
            for tool in layer.tools:
                if tool.name not in seen:
                    seen.add(tool.name)
                    result.append(tool)
        return result

    def all_documents(self) -> list[Document]:
        """Return documents from all DOCUMENT layers in insertion order."""
        return [
            layer.document for layer in self.layers if isinstance(layer, DocumentLayer)
        ]

    def all_bundles(self) -> list[ContentBundle]:
        """Return bundles from all CONTENT_BUNDLES layers in insertion order."""
        return [
            bundle
            for layer in self.layers
            if isinstance(layer, ContentBundlesLayer)
            for bundle in layer.bundles
        ]

    def layers_of_type(
        self,
        layer_types: LayerType | AnyContextLayer | list[LayerType | AnyContextLayer],
    ) -> list[AnyContextLayer]:
        items = layer_types if isinstance(layer_types, list) else [layer_types]
        normalized = {lt.type if isinstance(lt, BaseModel) else lt for lt in items}
        return [layer for layer in self.layers if layer.type in normalized]

    # ------------------------------------------------------------------
    # Immutable mutation helpers
    # ------------------------------------------------------------------
    def insert_layer(self, layer: AnyContextLayer, index: int) -> "ContextStack":
        """Return a new stack with *layer* inserted at *index* (default 0)."""
        return ContextStack(layers=[*self.layers[:index], layer, *self.layers[index:]])

    def append_layer(self, layer: AnyContextLayer) -> "ContextStack":
        """Return a new stack with *layer* appended."""
        return ContextStack(layers=[*self.layers, layer])

    def append_layers(self, layers: list[AnyContextLayer]) -> "ContextStack":
        """Return a new stack with *layers* appended."""
        return ContextStack(layers=[*self.layers, *layers])

    def remove_layers_of_type(self, layer_type: LayerType) -> "ContextStack":
        """Return a new stack with all layers of *layer_type* removed."""
        return ContextStack(
            layers=[layer for layer in self.layers if layer.type is not layer_type]
        )

    def remove_layers_of_source(self, source: str) -> "ContextStack":
        """Return a new stack with all layers of *source* removed."""
        return ContextStack(
            layers=[layer for layer in self.layers if layer.source != source]
        )
