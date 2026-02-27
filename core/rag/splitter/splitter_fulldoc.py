from typing import List, Callable
import copy

from core.rag.entities.document import Document
from core.rag.splitter.splitter_base import BaseSplitter
from core.rag.utils.rag_utils import get_text_id, get_text_hash, len_without_link

_ADVANCED_FULL_DOC_MAX_CHARACTERS = 10000

class FulldocSplitter(BaseSplitter):
    def __init__(self, length_function: Callable[[str], int] = len, **kwargs):
        if 'sonchunk_setting' not in kwargs:
            raise ValueError('FulldocSplitter: missing required parameter: "sonchunk_setting".')
        
        self.sonchunk_setting = kwargs['sonchunk_setting']
        self.sonchunk_setting['child'] = True
        self.length_function = length_function

    async def split_chunks(self, documents: List[Document]) -> List[Document]:
        # documents合并为一个document
        page_content = ''.join([document.page_content for document in documents])
        page_content = page_content[:_ADVANCED_FULL_DOC_MAX_CHARACTERS]
        father_document = Document(page_content=page_content, metadata=copy.deepcopy(documents[0].metadata))

        # 切分ChildDocument
        child_documents = self._split_chunks(
            documents=[copy.deepcopy(father_document)],
            length_function=self.length_function,
            pic_insert_lf=True,
            **(self.sonchunk_setting),
        )

        father_document.children = child_documents

        # 更新 father_document的page_content
        father_document.page_content = ''.join([child.page_content for child in father_document.children])

        father_document.metadata['content_id'] = get_text_id(father_document.page_content)
        father_document.metadata['content_hash'] = get_text_hash(father_document.page_content)
        father_document.metadata['content_len'] = len_without_link(father_document.page_content)
        for child in father_document.children:
            child.metadata['content_id'] = get_text_id(child.page_content)
            child.metadata['content_hash'] = get_text_hash(child.page_content)
            child.metadata['content_len'] = len_without_link(child.page_content)

        return [father_document]