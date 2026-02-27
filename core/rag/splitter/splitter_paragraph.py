from typing import List, Callable
import copy

from core.rag.entities.document import Document
from core.rag.splitter.splitter_base import BaseSplitter
from core.rag.utils.rag_utils import get_text_id, get_text_hash, len_without_link

class ParagraphSplitter(BaseSplitter):
    def __init__(self, length_function: Callable[[str], int] = len, **kwargs):
        if 'fatherchunk_setting' not in kwargs:
            raise ValueError('ParagraphSplitter: missing required parameter: "fatherchunk_setting".')
        if 'sonchunk_setting' not in kwargs:
            raise ValueError('ParagraphSplitter: missing required parameter: "sonchunk_setting".')
        
        self.fatherchunk_setting = kwargs['fatherchunk_setting']
        self.sonchunk_setting = kwargs['sonchunk_setting']
        self.sonchunk_setting['child'] = True
        self.length_function = length_function

    async def split_chunks(self, documents: List[Document]) -> List[Document]:
        # 父段切分
        father_documents = self._split_chunks(
            documents=documents,
            length_function=self.length_function,
            pic_insert_lf=False,            # 此处父段分段，不在图片前后插入换行，避免干扰子段切分
            **(self.fatherchunk_setting),
        )

        # 将切分的父段切分成子段
        for father_document in father_documents:
            child_documents = self._split_chunks(
                documents=[copy.deepcopy(father_document)],
                length_function=self.length_function,
                pic_insert_lf=True,
                **(self.sonchunk_setting),
            )
            father_document.children = child_documents

            # 更新 father_document的page_content
            father_document.page_content = ''.join([child.page_content for child in father_document.children])
        
        for father_document in father_documents:
            father_document.metadata['content_id'] = get_text_id(father_document.page_content)
            father_document.metadata['content_hash'] = get_text_hash(father_document.page_content)
            father_document.metadata['content_len'] = len_without_link(father_document.page_content)
            for child in father_document.children:
                child.metadata['content_id'] = get_text_id(child.page_content)
                child.metadata['content_hash'] = get_text_hash(child.page_content)
                child.metadata['content_len'] = len_without_link(child.page_content)

        return father_documents