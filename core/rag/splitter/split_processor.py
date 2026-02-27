from enum import StrEnum
from typing import List, Tuple, Optional

from core.rag.entities.document import Document
from core.rag.splitter.splitter_base import BaseSplitter
from core.rag.splitter.splitter_normal import NormalSplitter
from core.rag.splitter.splitter_qa import QaSplitter
from core.rag.splitter.splitter_paragraph import ParagraphSplitter
from core.rag.splitter.splitter_fulldoc import FulldocSplitter
from core.rag.utils.rag_utils import remove_leading_symbols
from core.rag.splitter.splitter_entities import SplitType

ADVANCED_FULL_DOC_MAX_CHARACTERS = 10000
NORMAL_QA_MAX_WORKER=4

_splitter_map = {
    SplitType.NORMAL: NormalSplitter,
    SplitType.NORMAL_QA: QaSplitter,
    SplitType.ADVANCED_FULL_DOC: FulldocSplitter,
    SplitType.ADVANCED_PARAGRAPH: ParagraphSplitter,
}

class SplitProcessor:

    @classmethod
    def _get_type(cls, **kwargs) -> SplitType:
        if kwargs.get('chunk_setting'):
            if kwargs.get('qa_setting', {}).get('enable'):
                return SplitType.NORMAL_QA
            else:
                return SplitType.NORMAL
        else:
            if not kwargs.get('fatherchunk_setting', {}).get('fulltext'):
                return SplitType.ADVANCED_PARAGRAPH
            else:
                return SplitType.ADVANCED_FULL_DOC

    @classmethod
    async def split(cls, documents: List[Document], **kwargs) -> Tuple[str, List[Document]]:
        '''
            根据切分类型，进行不同种类的切分
        '''

        split_type = cls._get_type(**kwargs)
        length_function = len
        splitter: Optional[BaseSplitter] = None

        if split_type not in _splitter_map:
            raise ValueError(f'Unsupported split type: {split_type.value}')
        
        splitter = _splitter_map[split_type](length_function=length_function, **kwargs)
        chunks = await splitter.split_chunks(documents)

        # for chunk in chunks:
        #     chunk.page_content = remove_leading_symbols(chunk.page_content)

        return split_type.value, chunks