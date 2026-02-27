from typing import List, Callable

from core.rag.entities.document import Document
from core.rag.splitter.splitter_base import BaseSplitter
from core.rag.utils.rag_utils import get_text_id, get_text_hash, len_without_link

class NormalSplitter(BaseSplitter):
    def __init__(self, length_function: Callable[[str], int] = len, **kwargs):
        if 'chunk_setting' not in kwargs:
            raise ValueError('NormalSplitter: missing required parameter: "chunk_setting".')
        
        self.chunk_setting = kwargs['chunk_setting']
        self.length_function = length_function

    async def split_chunks(self, documents: List[Document]) -> List[Document]:
        chunks = self._split_chunks(
            documents=documents,
            length_function=self.length_function,
            pic_insert_lf=True,
            **(self.chunk_setting),
        )

        for chunk in chunks:
            chunk.metadata['content_id'] = get_text_id(chunk.page_content)
            chunk.metadata['content_hash'] = get_text_hash(chunk.page_content)
            chunk.metadata['content_len'] = len_without_link(chunk.page_content)

        return chunks
    