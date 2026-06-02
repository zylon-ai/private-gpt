import enum

from llama_index.core.schema import BaseNode


class MetadataKeys(enum.StrEnum):
    """General metadata keys."""

    # General
    COLLECTION = "collection"
    FILE_HASH = "hash"

    # BE
    ARTIFACT_ID = "artifact_id"
    FILENAME = "file_name"
    FILETYPE = "file_type"
    PROJECT_ID = "project_id"
    ORGANIZATION_ID = "org_id"

    # Models
    LLM_MODEL = "llm_model"
    EMBED_MODEL = "embed_model"


class MetadataDocument(enum.StrEnum):
    """Metadata used to identify documents.

    These are the metadata keys that are used to identify documents.
    All of these keys are sent back to the client.
    """

    FILENAME = MetadataKeys.FILENAME.value
    FILETYPE = MetadataKeys.FILETYPE.value
    ARTIFACT_ID = MetadataKeys.ARTIFACT_ID.value
    COLLECTION = MetadataKeys.COLLECTION.value
    ORGANIZATION_ID = MetadataKeys.ORGANIZATION_ID.value
    PROJECT_ID = MetadataKeys.PROJECT_ID.value


class MetadataFeatures(enum.Enum):
    """Metadata used to identify features."""

    class MetadataSummaryFeature(enum.StrEnum):
        SUMMARY = "summary"

    class MetadataTableFeature(enum.StrEnum):

        TABLE_ID = "table_id"
        TABLE_HEADER = "table_header"
        TABLE_CONTENT = "table_content"
        TABLE_DESCRIPTION = "table_description"
        TABLE_RAW_CONTENT = "table_raw_content"


class MetadataNode(enum.StrEnum):
    """Metadata used to identify node information.

    These are the metadata keys that are used to identify node information.
    """

    TOKEN_COUNT = "token_count"
    SCORE = "score"
    CORRELATION_ID = "correlation_id"


class MetadataFlags(enum.StrEnum):
    """Metadata flags that are used to store boolean metadata."""

    SHORTER_ID = "shorter_id"
    NON_CITABLE = "non_citable"
    SKIP_RETURN = "skip_return"

    NO_PRUNABLE = "no_prunable"
    HIDDEN = "hidden"


class MetadataChunk(enum.StrEnum):
    """Metadata used to identify chunk information.

    These are the metadata keys that are used to identify chunk information.
    All of these keys are sent back to the client.
    """

    TITLE = "title"
    PAGE = "page"
    SHORTER_ID = MetadataFlags.SHORTER_ID.value
    ABS_IDX = "abs_idx"
    REL_IDX = "rel_idx"


class MetadataHelper:

    ##############################
    # Exclude metadata functions #
    ##############################

    @staticmethod
    def exclude_metadata(node: BaseNode) -> None:
        MetadataHelper.exclude_general_metadata(node)
        MetadataHelper.exclude_feature_metadata(node)
        MetadataHelper.exclude_flags_metadata(node)

    @staticmethod
    def exclude_general_metadata(node: BaseNode) -> None:
        MetadataHelper.exclude_key_metadata(
            node,
            MetadataKeys.ARTIFACT_ID.value,
            exclude_llm=True,
            exclude_from_embed=True,
        )
        MetadataHelper.exclude_key_metadata(
            node,
            MetadataKeys.COLLECTION.value,
            exclude_llm=True,
            exclude_from_embed=True,
        )
        MetadataHelper.exclude_key_metadata(
            node,
            MetadataKeys.FILE_HASH.value,
            exclude_llm=True,
            exclude_from_embed=True,
        )
        MetadataHelper.exclude_key_metadata(
            node,
            MetadataKeys.ARTIFACT_ID.value,
            exclude_llm=True,
            exclude_from_embed=True,
        )
        MetadataHelper.exclude_key_metadata(
            node,
            MetadataKeys.FILENAME.value,
            exclude_llm=True,
            exclude_from_embed=False,
        )
        MetadataHelper.exclude_key_metadata(
            node,
            MetadataKeys.FILETYPE.value,
            exclude_llm=True,
            exclude_from_embed=True,
        )
        MetadataHelper.exclude_key_metadata(
            node,
            MetadataKeys.ORGANIZATION_ID.value,
            exclude_llm=True,
            exclude_from_embed=True,
        )
        MetadataHelper.exclude_key_metadata(
            node,
            MetadataKeys.PROJECT_ID.value,
            exclude_llm=True,
            exclude_from_embed=True,
        )
        MetadataHelper.exclude_key_metadata(
            node,
            MetadataKeys.LLM_MODEL.value,
            exclude_llm=True,
            exclude_from_embed=True,
        )
        MetadataHelper.exclude_key_metadata(
            node,
            MetadataKeys.EMBED_MODEL.value,
            exclude_llm=True,
            exclude_from_embed=True,
        )
        MetadataHelper.exclude_key_metadata(
            node,
            MetadataChunk.PAGE,
            exclude_llm=True,
            exclude_from_embed=False,
        )
        MetadataHelper.exclude_key_metadata(
            node,
            MetadataChunk.PAGE,
            exclude_llm=True,
            exclude_from_embed=False,
        )

    @staticmethod
    def exclude_feature_metadata(node: BaseNode) -> None:
        features = list(  # type: ignore
            MetadataFeatures.MetadataTableFeature.value
        ) + list(  # type: ignore
            MetadataFeatures.MetadataSummaryFeature.value
        )
        for key in features:
            MetadataHelper.exclude_key_metadata(
                node, key, exclude_llm=True, exclude_from_embed=True
            )

    @staticmethod
    def exclude_flags_metadata(node: BaseNode) -> None:
        for key in MetadataFlags:
            MetadataHelper.exclude_key_metadata(
                node, key, exclude_llm=True, exclude_from_embed=True
            )

    @staticmethod
    def exclude_key_metadata(
        node: BaseNode, key: str, exclude_llm: bool, exclude_from_embed: bool
    ) -> None:
        if key in node.metadata:
            if exclude_llm and key not in node.excluded_llm_metadata_keys:
                node.excluded_llm_metadata_keys.append(key)
            if exclude_from_embed and key not in node.excluded_embed_metadata_keys:
                node.excluded_embed_metadata_keys.append(key)

    ################################
    # Remove metadata properties   #
    ################################
    @staticmethod
    def remove_metadata(node: BaseNode, key: str) -> None:
        if key not in node.metadata:
            return
        del node.metadata[key]
        if key in node.excluded_llm_metadata_keys:
            node.excluded_llm_metadata_keys.remove(key)
        if key in node.excluded_embed_metadata_keys:
            node.excluded_embed_metadata_keys.remove(key)

    ################################
    # Set metadata flags functions #
    ################################
    @staticmethod
    def set_non_citable(node: BaseNode) -> None:
        MetadataHelper._set_flag(node, MetadataFlags.NON_CITABLE.value)

    @staticmethod
    def set_non_returnable(node: BaseNode) -> None:
        MetadataHelper._set_flag(node, MetadataFlags.SKIP_RETURN.value)

    @staticmethod
    def _set_flag(node: BaseNode, key: str) -> None:
        node.metadata[key] = True
        MetadataHelper.exclude_key_metadata(
            node, key, exclude_llm=True, exclude_from_embed=True
        )
