from pathlib import Path
from typing import List, Optional, Dict

from bs4 import BeautifulSoup
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

class HTMLParser(BaseReader):
    """HTML parser."""

    def __init__(self) -> None:
        """Init parser."""
        super().__init__()

    def load_data(self, input_file: Path, extra_info: Optional[Dict] = {}) -> List[Document]:
        """Parse file."""
        documents = []  # This will contain the list of Document objects

        with open(input_file, "rb") as fp:
            soup = BeautifulSoup(fp, 'html.parser')
            text = soup.get_text()
            text = text.strip() if text else ''

            # Create a Document object with the extracted text and any extra metadata
            document = Document(text=text, metadata=extra_info)
            documents.append(document)

        return documents