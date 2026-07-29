from llama_index.core.base.llms.types import TextBlock
from pydantic import BaseModel, ConfigDict, Field

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.context.errors import NonResumableToolError
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

    def checkpoint_dump(self) -> dict[str, object]:
        """Serialize the complete stack after validating server tool restoration."""
        for layer in self.layers:
            if not isinstance(layer, ToolDefinitionsLayer):
                continue
            for tool in layer.tools:
                if tool.runtime == "server" and tool.execution_metadata is None:
                    raise NonResumableToolError(
                        f"Server tool {tool.name!r} from layer {layer.source!r} "
                        "does not define execution metadata."
                    )
        return self.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def to_system_prompt(self) -> list[TextBlock]:
        """Render prompt layers by priority (then insertion order).

        Layers stay isolated in the stack; deduplication happens only here, at
        render time. Two cases must be covered so the LLM never receives the
        same text twice:

        - *Stale duplicate*: a layer's text is reproduced verbatim by another
          kept layer (e.g. the chat header rendered both as a
          ``RuntimeInstructionsLayer`` and baked back into a re-ingested
          ``UserInstructionsLayer``). The later candidate is dropped because a
          kept block already contains it.
        - *Snowballed aggregate*: a re-ingested ``UserInstructionsLayer`` may
          embed the header, platform guidelines and even prior skill bodies
          that the interceptors regenerate, this iteration, as isolated
          layers. The freshly-generated isolated layers carry the latest
          state (e.g. the ``ContextPromptLayer`` rebuilt from the latest
          documents accumulated across iterations), so they must survive.
          The aggregate layer is the one discarded: its rendered text *is a
          superset of* (contains) one or more kept isolated blocks.

        Rendering therefore walks layers in *descending* priority order — the
        freshly-generated isolated layers (highest priority numbers) are kept
        first — and a candidate is dropped when its text contains, or is
        contained in, the text of any already-kept block. That preserves the
        latest isolated layers across iterations and discards the duplicate /
        stale aggregate layer at render, restoring the prompt to its original
        isolated shape.
        """
        ordered_layers = sorted(
            enumerate(self.layers),
            key=lambda item: (-item[1].priority, item[0]),
        )
        blocks: list[TextBlock] = []
        kept: list[str] = []
        for _, layer in ordered_layers:
            rendered = layer.render()
            if not rendered or not rendered.strip():
                continue
            text = rendered.strip()
            # Drop the candidate if its text already appears inside a kept
            # block (stale duplicate) OR if it reproduces, as a snowballed
            # aggregate, any kept isolated block. The freshly-generated
            # isolated layers are kept first thanks to the descending-priority
            # order, so the aggregate (e.g. a re-ingested UserInstructions
            # layer built from a previous response) is the one discarded,
            # while the latest isolated layers — including the ContextPrompt
            # rebuilt with the latest documents — survive.
            is_stale_duplicate = any(text in candidate for candidate in kept)
            is_snowballed_aggregate = any(candidate in text for candidate in kept)
            if is_stale_duplicate or is_snowballed_aggregate:
                continue
            blocks.append(TextBlock(text=text))
            kept.append(text)
        return blocks

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

    def all_bundles_to_remove(self) -> list[str]:
        """Return canonical paths to remove from all CONTENT_BUNDLES layers."""
        return [
            path
            for layer in self.layers
            if isinstance(layer, ContentBundlesLayer)
            for path in layer.to_remove
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
    def _remove_duplicate(self, layer: AnyContextLayer) -> "list[AnyContextLayer]":
        """Return current layers with any existing (type, source) duplicate removed."""
        return [
            existing
            for existing in self.layers
            if not (existing.type == layer.type and existing.source == layer.source)
        ]

    def insert_layer(self, layer: AnyContextLayer, index: int) -> "ContextStack":
        """Return a new stack with *layer* inserted at *index* (default 0)."""
        base = self._remove_duplicate(layer)
        index = min(index, len(base))
        return ContextStack(layers=[*base[:index], layer, *base[index:]])

    def append_layer(self, layer: AnyContextLayer) -> "ContextStack":
        """Return a new stack with *layer* appended."""
        return ContextStack(layers=[*self._remove_duplicate(layer), layer])

    def append_layers(self, layers: list[AnyContextLayer]) -> "ContextStack":
        """Return a new stack with *layers* appended."""
        stack = self
        for layer in layers:
            stack = stack.append_layer(layer)
        return stack

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
