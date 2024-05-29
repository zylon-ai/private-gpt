from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.indices import SimpleKeywordTableIndex

class VectorStoreIndex1(VectorStoreIndex):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyword_index = None

    def set_keyword_index(self, keyword_index: SimpleKeywordTableIndex):
        self.keyword_index = keyword_index

    def get_keyword_index(self) -> SimpleKeywordTableIndex:
        return self.keyword_index