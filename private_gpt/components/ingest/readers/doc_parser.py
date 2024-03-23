import subprocess
from pathlib import Path
from typing import List, Optional, Dict

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

class DOCParser(BaseReader):
    """DOC parser."""

    def __init__(self) -> None:
        """Init parser."""
        super().__init__()

    def load_data(self, input_file: Path, extra_info: Optional[Dict] = {}) -> List[Document]:
        """Parse file."""
        documents = []  # This will contain the list of Document objects

        # Call antiword to convert the .doc file to plain text
        try:
            text = subprocess.run(['antiword', input_file], capture_output=True, text=True, check=True).stdout
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {input_file}: {e}")
            text = ""

        # Create a Document object with the extracted text and any extra metadata
        if text:
            document = Document(text=text, metadata=extra_info)
            documents.append(document)

        return documents