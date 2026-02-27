from pathlib import Path
from typing import Optional, List
from uuid import uuid4

from config.config import get_config
from core.rag.entities.document import Document
from core.rag.extractor.extractor_base import BaseExtractor
from core.rag.extractor.structured.excel_extractor import ExcelExtractor
from core.rag.extractor.structured.pdf_extractor import PdfExtractor
from core.rag.extractor.structured.markdown_extractor import MarkdownExtractor
from core.rag.extractor.structured.html_extractor import HtmlExtractor
from core.rag.extractor.structured.word_extractor import WordExtractor
from core.rag.extractor.structured.csv_extractor import CSVExtractor
from core.rag.extractor.structured.text_extractor import TextExtractor

class ExtractProcessor:

    @classmethod
    def extract(cls, file_path: Path, file_id: str) -> List[Document]:
        file_extension = file_path.suffix.lower()
        file_path = str(file_path)
        extractor: Optional[BaseExtractor] = None

        if file_extension in ['.xlsx']:
            extractor = ExcelExtractor(file_path)
        elif file_extension in ['.md', '.mdx', '.markdown']:
            extractor = MarkdownExtractor(file_path, autodetect_encoding=True)
        elif file_extension in ['.htm', '.html']:
            extractor = HtmlExtractor(file_path)
        elif file_extension in ['.pdf']:
            extractor = PdfExtractor(file_path, file_id=file_id)
        elif file_extension in ['.docx']:
            extractor = WordExtractor(file_path, file_id=file_id)
        elif file_extension in ['.csv']:
            extractor = CSVExtractor(file_path, autodetect_encoding=True)
        elif file_extension in ['.txt']:
            extractor = TextExtractor(file_path, autodetect_encoding=True)
        else:
            raise RuntimeError(f'Unsupported file extension: {file_extension}')

        documents = extractor.extract()
        return documents
