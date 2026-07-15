"""Test validation logic for model cache.

Scenario 8: Validation Edge Cases
"""

from pathlib import Path

from private_gpt.components.llm.tokenizers.models.model_cache import (
    has_all_safetensors,
    has_tokenizer_files,
    validate_model_path,
)


class TestTokenizerFileDetection:
    """Test has_tokenizer_files() with various tokenizer formats."""

    def test_standard_hf_tokenizer(
        self, hf_cache_dir: Path, tokenizer_files: list[str]
    ):
        """Standard HF tokenizer (tokenizer.json + tokenizer_config.json) → True."""
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        for filename in tokenizer_files:
            (test_dir / filename).touch()

        assert has_tokenizer_files(test_dir) is True

    def test_mistral_tokenizer(
        self, hf_cache_dir: Path, mistral_tokenizer_files: list[str]
    ):
        """Mistral tekken tokenizer (tekken.json) → True."""
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        for filename in mistral_tokenizer_files:
            (test_dir / filename).touch()

        assert has_tokenizer_files(test_dir) is True

    def test_incomplete_hf_tokenizer_missing_config(self, hf_cache_dir: Path):
        """Only tokenizer.json (missing tokenizer_config.json) → False."""
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        (test_dir / "tokenizer.json").touch()

        assert has_tokenizer_files(test_dir) is False

    def test_incomplete_hf_tokenizer_missing_json(self, hf_cache_dir: Path):
        """Only tokenizer_config.json (missing tokenizer.json) → False."""
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        (test_dir / "tokenizer_config.json").touch()

        assert has_tokenizer_files(test_dir) is False

    def test_empty_directory(self, hf_cache_dir: Path):
        """Empty directory → False."""
        test_dir = hf_cache_dir / "empty"
        test_dir.mkdir()

        assert has_tokenizer_files(test_dir) is False

    def test_both_hf_and_mistral_tokenizers(self, hf_cache_dir: Path):
        """Both HF and Mistral tokenizers present → True."""
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        (test_dir / "tokenizer.json").touch()
        (test_dir / "tokenizer_config.json").touch()
        (test_dir / "tekken.json").touch()

        assert has_tokenizer_files(test_dir) is True


class TestSafetensorsDetection:
    """Test has_safetensors() for model weight detection."""

    def test_single_safetensors_file(self, hf_cache_dir: Path):
        """Directory with single .safetensors file → True."""
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        (test_dir / "model.safetensors").touch()

        assert has_all_safetensors(test_dir) is True

    def test_multiple_safetensors_files(
        self, hf_cache_dir: Path, safetensors_files: list[str]
    ):
        """Directory with multiple .safetensors files → True."""
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        for filename in safetensors_files:
            (test_dir / filename).touch()

        assert has_all_safetensors(test_dir) is True

    def test_no_safetensors(self, hf_cache_dir: Path):
        """Directory without .safetensors files → False."""
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        (test_dir / "model.bin").touch()
        (test_dir / "config.json").touch()

        assert has_all_safetensors(test_dir) is False

    def test_empty_directory(self, hf_cache_dir: Path):
        """Empty directory → False."""
        test_dir = hf_cache_dir / "empty"
        test_dir.mkdir()

        assert has_all_safetensors(test_dir) is False


class TestValidateModelPath:
    """Test validate_model_path() with tokenizer_only flag.

    Scenario 8.1, 8.2: Full model vs tokenizer-only validation.
    """

    def test_tokenizer_only_with_tokenizer_files(
        self, hf_cache_dir: Path, tokenizer_files: list[str]
    ):
        """tokenizer_only=True: Tokenizer files, no safetensors → True."""
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        for filename in tokenizer_files:
            (test_dir / filename).touch()

        assert validate_model_path(test_dir, tokenizer_only=True) is True

    def test_tokenizer_only_with_full_model(
        self,
        hf_cache_dir: Path,
        tokenizer_files: list[str],
        safetensors_files: list[str],
    ):
        """tokenizer_only=True: Full model (tokenizer + safetensors) → True.

        Scenario 8.1: Tokenizer-only request finds full model.
        """
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        for filename in tokenizer_files + safetensors_files:
            (test_dir / filename).touch()

        # Superset satisfies subset requirement
        assert validate_model_path(test_dir, tokenizer_only=True) is True

    def test_tokenizer_only_without_tokenizer(self, hf_cache_dir: Path):
        """tokenizer_only=True: Only safetensors, no tokenizer → False."""
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        (test_dir / "model.safetensors").touch()

        assert validate_model_path(test_dir, tokenizer_only=True) is False

    def test_full_model_with_both(
        self,
        hf_cache_dir: Path,
        tokenizer_files: list[str],
        safetensors_files: list[str],
    ):
        """tokenizer_only=False: Directory with tokenizer AND safetensors → True."""
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        for filename in tokenizer_files + safetensors_files:
            (test_dir / filename).touch()

        assert validate_model_path(test_dir, tokenizer_only=False) is True

    def test_full_model_only_tokenizer(
        self, hf_cache_dir: Path, tokenizer_files: list[str]
    ):
        """tokenizer_only=False: Directory with only tokenizer → False .

        Scenario 8.2: Full model request finds only tokenizer.
        """
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        for filename in tokenizer_files:
            (test_dir / filename).touch()

        # Subset doesn't satisfy full requirement
        assert validate_model_path(test_dir, tokenizer_only=False) is False

    def test_full_model_only_safetensors(
        self, hf_cache_dir: Path, safetensors_files: list[str]
    ):
        """tokenizer_only=False: Directory with only safetensors → False."""
        test_dir = hf_cache_dir / "test_model"
        test_dir.mkdir()
        for filename in safetensors_files:
            (test_dir / filename).touch()

        assert validate_model_path(test_dir, tokenizer_only=False) is False

    def test_empty_directory(self, hf_cache_dir: Path):
        """Both flags: Empty directory → False."""
        test_dir = hf_cache_dir / "empty"
        test_dir.mkdir()

        assert validate_model_path(test_dir, tokenizer_only=True) is False
        assert validate_model_path(test_dir, tokenizer_only=False) is False

    def test_nonexistent_path(self, hf_cache_dir: Path):
        """Both flags: Non-existent path → False."""
        test_dir = hf_cache_dir / "nonexistent"

        assert validate_model_path(test_dir, tokenizer_only=True) is False
        assert validate_model_path(test_dir, tokenizer_only=False) is False

    def test_file_not_directory(self, hf_cache_dir: Path):
        """Both flags: File (not directory) → False."""
        test_file = hf_cache_dir / "test_file.txt"
        test_file.touch()

        assert validate_model_path(test_file, tokenizer_only=True) is False
        assert validate_model_path(test_file, tokenizer_only=False) is False

    def test_mistral_tokenizer_recognized(
        self, hf_cache_dir: Path, mistral_tokenizer_files: list[str]
    ):
        """Scenario 8.3: Mistral tokenizer (tekken.json) recognized."""
        test_dir = hf_cache_dir / "mistral_model"
        test_dir.mkdir()
        for filename in mistral_tokenizer_files:
            (test_dir / filename).touch()

        # Mistral tokenizer should be valid for tokenizer_only
        assert validate_model_path(test_dir, tokenizer_only=True) is True

        # But not for full model (needs safetensors)
        assert validate_model_path(test_dir, tokenizer_only=False) is False
