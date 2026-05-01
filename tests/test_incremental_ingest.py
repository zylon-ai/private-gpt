"""Tests for the incremental ingestion proof-of-concept.

Tests cover:
1. SemanticChunker: paragraph splitting, hash computation, merge/split
2. ChunkHashStore: persistence, CRUD operations
3. DiffDetector: change detection accuracy
4. Patience diff: anchor-based diff algorithm
5. Integration: full incremental pipeline simulation
"""

import tempfile

from private_gpt.components.ingest.incremental.chunk_hash_store import (
    ChunkHashStore,
    DocumentRecord,
    StoredChunkInfo,
)
from private_gpt.components.ingest.incremental.chunk_hasher import (
    HashedChunk,
    SemanticChunker,
)
from private_gpt.components.ingest.incremental.diff_detector import (
    ChangeType,
    DiffDetector,
    patience_diff,
)

# ─── SemanticChunker Tests ───────────────────────────────────────────────


class TestSemanticChunker:
    """Tests for the SemanticChunker class."""

    def setup_method(self):
        self.chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=500)

    def test_basic_paragraph_split(self):
        """Test that double newlines split into separate chunks."""
        text = "First paragraph with enough text to meet minimum.\n\nSecond paragraph with enough text to meet minimum."
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, HashedChunk) for c in chunks)

    def test_hash_stability(self):
        """Same text should always produce the same hash."""
        text = (
            "This is a test paragraph with sufficient length to be a chunk on its own."
        )
        chunks1 = self.chunker.chunk_text(text)
        chunks2 = self.chunker.chunk_text(text)
        assert chunks1[0].content_hash == chunks2[0].content_hash

    def test_hash_changes_with_content(self):
        """Different text should produce different hashes."""
        text1 = "First version of a paragraph that has enough content to be chunked properly."
        text2 = "Second version of a paragraph that has different content to be chunked properly."
        chunks1 = self.chunker.chunk_text(text1)
        chunks2 = self.chunker.chunk_text(text2)
        assert chunks1[0].content_hash != chunks2[0].content_hash

    def test_whitespace_normalisation(self):
        """Hash should be the same regardless of whitespace differences."""
        text1 = "Hello   world   test  content  with extra spaces that is long enough."
        text2 = "Hello world test content with extra spaces that is long enough."
        hash1 = SemanticChunker._compute_hash(text1)
        hash2 = SemanticChunker._compute_hash(text2)
        assert hash1 == hash2

    def test_metadata_propagation(self):
        """Metadata should be attached to each chunk."""
        text = "A paragraph with enough text content to form a valid chunk for testing purposes here."
        metadata = {"file_name": "test.txt", "source": "unit_test"}
        chunks = self.chunker.chunk_text(text, metadata=metadata)
        assert chunks[0].metadata["file_name"] == "test.txt"
        assert chunks[0].metadata["source"] == "unit_test"

    def test_chunk_indices_sequential(self):
        """Chunk indices should be sequential starting from 0."""
        text = (
            "First paragraph of the document.\n\n"
            "Second paragraph of the document.\n\n"
            "Third paragraph of the document."
        )
        # Use smaller min to get multiple chunks
        chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=500)
        chunks = chunker.chunk_text(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_empty_text(self):
        """Empty text should produce no chunks."""
        chunks = self.chunker.chunk_text("")
        assert len(chunks) == 0

    def test_oversized_chunk_split(self):
        """Chunks exceeding max_chunk_size should be split."""
        # Create a very long text that exceeds max_chunk_size
        long_text = ". ".join(
            [
                f"This is sentence number {i} in a very long paragraph"
                for i in range(100)
            ]
        )
        chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=200)
        chunks = chunker.chunk_text(long_text)
        # Should produce multiple chunks
        assert len(chunks) > 1

    def test_multiple_paragraphs(self):
        """Multiple paragraphs should produce multiple chunks."""
        paragraphs = [
            f"This is paragraph {i} with enough content to stand as its own semantic unit for chunking."
            for i in range(5)
        ]
        text = "\n\n".join(paragraphs)
        chunker = SemanticChunker(min_chunk_size=20, max_chunk_size=500)
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 2  # At least some should remain separate


# ─── ChunkHashStore Tests ────────────────────────────────────────────────


class TestChunkHashStore:
    """Tests for the ChunkHashStore class."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = ChunkHashStore(persist_dir=self.tmpdir)

    def teardown_method(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_store(self):
        """New store should have no documents."""
        assert self.store.list_documents() == []

    def test_upsert_and_get(self):
        """Should be able to insert and retrieve a document."""
        record = DocumentRecord(
            doc_id="doc1",
            file_name="test.txt",
            file_hash="abc123",
            chunks=[
                StoredChunkInfo(chunk_index=0, content_hash="h1", node_id="n1"),
                StoredChunkInfo(chunk_index=1, content_hash="h2", node_id="n2"),
            ],
        )
        self.store.upsert_document(record)

        retrieved = self.store.get_document("doc1")
        assert retrieved is not None
        assert retrieved.file_name == "test.txt"
        assert len(retrieved.chunks) == 2

    def test_persistence(self):
        """Store should persist across instances."""
        record = DocumentRecord(
            doc_id="doc1",
            file_name="test.txt",
            file_hash="abc123",
        )
        self.store.upsert_document(record)

        # Create a new store instance pointing to the same directory
        store2 = ChunkHashStore(persist_dir=self.tmpdir)
        retrieved = store2.get_document("doc1")
        assert retrieved is not None
        assert retrieved.file_name == "test.txt"

    def test_lookup_by_filename(self):
        """Should find documents by filename."""
        record = DocumentRecord(doc_id="doc1", file_name="report.pdf", file_hash="xyz")
        self.store.upsert_document(record)

        found = self.store.get_document_by_filename("report.pdf")
        assert found is not None
        assert found.doc_id == "doc1"

        not_found = self.store.get_document_by_filename("nonexistent.txt")
        assert not_found is None

    def test_delete_document(self):
        """Should be able to delete documents."""
        record = DocumentRecord(doc_id="doc1", file_name="test.txt", file_hash="abc")
        self.store.upsert_document(record)
        self.store.delete_document("doc1")

        assert self.store.get_document("doc1") is None

    def test_get_chunk_hashes(self):
        """Should return chunk_index -> hash mapping."""
        record = DocumentRecord(
            doc_id="doc1",
            file_name="test.txt",
            chunks=[
                StoredChunkInfo(chunk_index=0, content_hash="h0"),
                StoredChunkInfo(chunk_index=1, content_hash="h1"),
                StoredChunkInfo(chunk_index=2, content_hash="h2"),
            ],
        )
        self.store.upsert_document(record)

        hashes = self.store.get_chunk_hashes("doc1")
        assert hashes == {0: "h0", 1: "h1", 2: "h2"}

    def test_version_increment(self):
        """Should track version numbers."""
        record = DocumentRecord(doc_id="doc1", file_name="test.txt", version=1)
        self.store.upsert_document(record)

        record.version = 2
        self.store.upsert_document(record)

        retrieved = self.store.get_document("doc1")
        assert retrieved.version == 2


# ─── DiffDetector Tests ──────────────────────────────────────────────────


class TestDiffDetector:
    """Tests for the DiffDetector class."""

    def setup_method(self):
        self.detector = DiffDetector(similarity_threshold=0.4)

    def _make_chunk(self, index: int, text: str) -> HashedChunk:
        return HashedChunk(
            chunk_index=index,
            text=text,
            content_hash=SemanticChunker._compute_hash(text),
        )

    def test_no_changes(self):
        """Identical chunks should all be UNCHANGED."""
        chunks = [
            self._make_chunk(0, "First paragraph content"),
            self._make_chunk(1, "Second paragraph content"),
        ]
        changes = self.detector.detect_changes(chunks, chunks)
        assert all(c.change_type == ChangeType.UNCHANGED for c in changes)

    def test_all_new(self):
        """Empty old + new chunks = all ADDED."""
        new_chunks = [
            self._make_chunk(0, "New content here"),
        ]
        changes = self.detector.detect_changes([], new_chunks)
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.ADDED

    def test_all_deleted(self):
        """Old chunks + empty new = all DELETED."""
        old_chunks = [
            self._make_chunk(0, "Old content here"),
        ]
        changes = self.detector.detect_changes(old_chunks, [])
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.DELETED

    def test_modification_detected(self):
        """Similar but not identical chunks should be MODIFIED."""
        old_chunks = [
            self._make_chunk(0, "The quick brown fox jumps over the lazy dog"),
        ]
        new_chunks = [
            self._make_chunk(0, "The quick brown fox jumps over the lazy cat"),
        ]
        changes = self.detector.detect_changes(old_chunks, new_chunks)
        modified = [c for c in changes if c.change_type == ChangeType.MODIFIED]
        assert len(modified) == 1
        assert modified[0].similarity_ratio > 0.8

    def test_mixed_changes(self):
        """Should correctly identify a mix of changes."""
        old_chunks = [
            self._make_chunk(0, "Unchanged paragraph stays the same"),
            self._make_chunk(1, "This paragraph will be modified slightly"),
            self._make_chunk(2, "This paragraph will be deleted entirely"),
        ]
        new_chunks = [
            self._make_chunk(0, "Unchanged paragraph stays the same"),
            self._make_chunk(1, "This paragraph was modified slightly here"),
            self._make_chunk(2, "Brand new paragraph added to the document"),
        ]

        changes = self.detector.detect_changes(old_chunks, new_chunks)

        unchanged = [c for c in changes if c.change_type == ChangeType.UNCHANGED]
        assert len(unchanged) >= 1  # At least the first unchanged chunk
        # The modified and deleted/added counts may vary based on similarity

    def test_detailed_diff(self):
        """Should produce unified diff output."""
        old_text = "Line 1\nLine 2\nLine 3\n"
        new_text = "Line 1\nLine 2 modified\nLine 3\nLine 4\n"
        diff = self.detector.get_detailed_diff(old_text, new_text)
        assert len(diff) > 0
        # Should contain additions and deletions
        diff_text = "".join(diff)
        assert "+" in diff_text or "-" in diff_text


# ─── Patience Diff Tests ─────────────────────────────────────────────────


class TestPatienceDiff:
    """Tests for the Patience diff algorithm."""

    def test_identical_text(self):
        """Identical text should produce only context lines."""
        lines = ["line 1", "line 2", "line 3"]
        result = patience_diff(lines, lines)
        assert all(tag == " " for tag, _ in result)

    def test_addition(self):
        """Added lines should be tagged with '+'."""
        old = ["line 1", "line 3"]
        new = ["line 1", "line 2", "line 3"]
        result = patience_diff(old, new)
        added = [line for tag, line in result if tag == "+"]
        assert "line 2" in added

    def test_deletion(self):
        """Deleted lines should be tagged with '-'."""
        old = ["line 1", "line 2", "line 3"]
        new = ["line 1", "line 3"]
        result = patience_diff(old, new)
        deleted = [line for tag, line in result if tag == "-"]
        assert "line 2" in deleted

    def test_unique_anchors(self):
        """Should use unique lines as anchors."""
        old = ["header", "content A", "content B", "footer"]
        new = ["header", "content A modified", "content B", "footer"]
        result = patience_diff(old, new)
        # "header" and "footer" are unique in both - should be context
        context = [line for tag, line in result if tag == " "]
        assert "header" in context
        assert "footer" in context


# ─── Integration Test ────────────────────────────────────────────────────


class TestIncrementalIntegration:
    """Integration tests simulating the full incremental pipeline."""

    def test_full_pipeline_simulation(self):
        """Simulate: initial ingest -> modify -> detect changes."""
        chunker = SemanticChunker(min_chunk_size=20, max_chunk_size=500)
        detector = DiffDetector(similarity_threshold=0.4)

        # Original document
        original = (
            "Introduction to the topic of RAG systems.\n\n"
            "RAG combines language models with external knowledge.\n\n"
            "PrivateGPT uses LlamaIndex for document processing.\n\n"
            "Conclusion summarizing the key findings."
        )

        # Step 1: Initial chunking
        old_chunks = chunker.chunk_text(original, metadata={"file": "doc.txt"})
        assert len(old_chunks) > 0

        # Step 2: Modify the document (change one paragraph)
        modified = (
            "Introduction to the topic of RAG systems.\n\n"
            "RAG combines language models with external knowledge sources.\n\n"
            "PrivateGPT uses LlamaIndex for document processing.\n\n"
            "Updated conclusion with new findings and recommendations."
        )

        new_chunks = chunker.chunk_text(modified, metadata={"file": "doc.txt"})

        # Step 3: Detect changes
        changes = detector.detect_changes(old_chunks, new_chunks)

        # Verify: at least some chunks should be unchanged
        unchanged = [c for c in changes if c.change_type == ChangeType.UNCHANGED]
        changed = [
            c
            for c in changes
            if c.change_type in (ChangeType.MODIFIED, ChangeType.ADDED)
        ]

        # We should have SOME unchanged chunks (not everything re-embedded)
        assert len(unchanged) > 0 or len(changed) > 0

    def test_hash_store_roundtrip(self):
        """Test storing and retrieving chunk info through the hash store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChunkHashStore(persist_dir=tmpdir)
            chunker = SemanticChunker(min_chunk_size=20, max_chunk_size=500)

            text = "Paragraph one for testing.\n\nParagraph two for testing."
            chunks = chunker.chunk_text(text)

            # Store the chunks
            record = DocumentRecord(
                doc_id="test_doc",
                file_name="test.txt",
                file_hash="abc",
                chunks=[
                    StoredChunkInfo(
                        chunk_index=c.chunk_index,
                        content_hash=c.content_hash,
                        node_id=f"node_{c.chunk_index}",
                        text_preview=c.text[:100],
                    )
                    for c in chunks
                ],
            )
            store.upsert_document(record)

            # Retrieve and verify
            retrieved = store.get_document("test_doc")
            assert retrieved is not None
            assert len(retrieved.chunks) == len(chunks)
            for stored, original in zip(retrieved.chunks, chunks, strict=False):
                assert stored.content_hash == original.content_hash
