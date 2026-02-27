"""Abstract interface for document loader implementations."""

from collections.abc import Iterator
from typing import Optional, cast
import os
from pathlib import Path
import uuid

from core.rag.entities.blob import Blob
from core.rag.extractor.extractor_base import BaseExtractor
from core.rag.entities.document import Document
from core.rag.extractor.extractor_utils import save_image
from config.config import get_config
# from extensions.ext_storage import storage


class PdfExtractor(BaseExtractor):
    """Load pdf files.


    Args:
        file_path: Path to the file to load.
    """

    def __init__(self, file_path: str, file_id: str, file_cache_key: Optional[str] = None):
        """Initialize with file path."""
        self._file_path = file_path
        self._file_id = file_id
        self._file_cache_key = file_cache_key
        self.file_path_id = os.path.splitext(os.path.basename(self._file_path))[0]

        self.file_url = get_config().get('dependent_info').get('knowledge_base').get('file_url')
        self.pic_save_path_prefix = get_config().get('dependent_info').get('knowledge_base').get('pic_save_path_prefix')
        self.pic_url_prefix = get_config().get('dependent_info').get('knowledge_base').get('pic_url_prefix')
        if self.file_url.endswith('/'):
            self.file_url = self.file_url[:-1]
        if self.pic_url_prefix.endswith('/'):
            self.pic_url_prefix = self.pic_url_prefix[:-1]
        self.pic_save_path_prefix = Path(self.pic_save_path_prefix) / self.file_path_id
        os.makedirs(self.pic_save_path_prefix, exist_ok=True)

    def extract(self) -> list[Document]:
        # plaintext_file_exists = False
        # if self._file_cache_key:
        #     try:
        #         text = cast(bytes, storage.load(self._file_cache_key)).decode("utf-8")
        #         plaintext_file_exists = True
        #         return [Document(page_content=text)]
        #     except FileNotFoundError:
        #         pass
        documents = list(self._load())
        # text_list = []
        # for document in documents:
        #     text_list.append(document.page_content)
        # text = "\n\n".join(text_list)

        # # save plaintext file for caching
        # if not plaintext_file_exists and self._file_cache_key:
        #     storage.save(self._file_cache_key, text.encode("utf-8"))

        return documents

    def load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        blob = Blob.from_path(self._file_path)
        yield from self.parse(blob)

    def parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdfium2  # type: ignore

        with blob.as_bytes_io() as file_path:
            pdf_reader = pypdfium2.PdfDocument(file_path, autoclose=True)
            try:
                for page_number, page in enumerate(pdf_reader):
                    text_page = page.get_textpage()
                    content = text_page.get_text_range()
                    text_page.close()
                    page.close()
                    metadata = {"source": blob.source, "page": page_number}
                    yield Document(page_content=content, metadata=metadata)
            finally:
                pdf_reader.close()

    def _load(self,) -> list[Document]:
        import fitz
        doc = fitz.open(self._file_path)
        results = []
        
        for page_index in range(len(doc)):
            page = doc[page_index]
            text_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, "text", block_no, ...)
            text_blocks.sort(key=lambda b: b[1])   # 按 y 坐标排序，模拟阅读顺序

            # 抽取图像
            image_markdowns = []
            image_infos = page.get_images(full=True)
            for img_idx, img in enumerate(image_infos):
                xref = img[0]
                # 提取图像数据
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_id = str(uuid.uuid4())
                image_pure_name = image_id + '.' + image_ext
                image_path = self.pic_save_path_prefix / image_pure_name
                save_image(str(image_path), image_bytes)

                # 查找图像 bbox 位置
                y_position = img[7][5] if isinstance(img[7], (list, tuple)) and len(img[7]) >= 6 else 0

                markdown = f"![image]({self.file_url}{self.pic_url_prefix}/{self.file_path_id}/{image_pure_name})"
                image_markdowns.append((y_position, markdown))

            # 合并文本和图像
            combined = []
            img_idx = 0
            image_markdowns.sort()

            for block in text_blocks:
                block_y = block[1]
                # 在该文本前插入所有低于该Y位置的图片
                while img_idx < len(image_markdowns) and image_markdowns[img_idx][0] <= block_y:
                    combined.append(image_markdowns[img_idx][1])
                    img_idx += 1
                combined.append(block[4].strip())

            # 剩下图片放后面
            while img_idx < len(image_markdowns):
                combined.append(image_markdowns[img_idx][1])
                img_idx += 1

            page_markdown = "".join(combined)
            result = Document(page_content=page_markdown, metadata={'source': self._file_path, 'page': page_index})
            results.append(result)
        return results