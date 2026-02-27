from typing import List, Callable, Optional
import re
import copy
import asyncio
import itertools
import traceback
from threading import Event

from core.rag.entities.document import Document
from core.rag.splitter.splitter_base import BaseSplitter
from core.rag.utils.rag_utils import get_text_id, get_text_hash, len_without_link
from core.generator.qa_generator.qa_generator import agenerate_qa
from core.model.model_manager import ModelManager, ModelInstanceType
from logger import get_logger

logger = get_logger('rag')

_NORMAL_QA_MAX_WORKER = 1
_IMG_RE = re.compile(r"!\[.*?\]\((https?://[^\s)]+)\)")

class QaSplitter(BaseSplitter):
    def __init__(
        self, 
        length_function: Callable[[str], int] = len, 
        task_id: Optional[str] = None,
        **kwargs
    ):
        if 'chunk_setting' not in kwargs:
            raise ValueError('QaSplitter: missing required parameter: "chunk_setting".')
        if 'qa_setting' not in kwargs:
            raise ValueError('QaSplitter: missing required parameter: "qa_setting".')
        
        self.chunk_setting = kwargs['chunk_setting']
        self.qa_setting = kwargs['qa_setting']
        self.length_function = length_function
        self.task_id = task_id

        if 'language' not in self.qa_setting:
            raise ValueError('QaSplitter: missing required parameter: "language".')
        if 'model_instance_provider' not in self.qa_setting:
            raise ValueError('QaSplitter: missing required parameter: "model_instance_provider".')
        if 'model_instance_config' not in self.qa_setting:
            raise ValueError('QaSplitter: missing required parameter: "model_instance_config".')

        self.language = self.qa_setting['language']
        
        model_instance_provider = self.qa_setting['model_instance_provider']
        model_instance_config = self.qa_setting['model_instance_config']
        self.model_instance = ModelManager.get_model_instance(
            provider=model_instance_provider,
            model_type=ModelInstanceType.LLM,
            **model_instance_config
        )

    async def split_chunks(self, documents: List[Document]) -> List[Document]:

        # 先普通分段
        documents = self._split_chunks(
            documents=documents,
            length_function=self.length_function,
            pic_insert_lf=True,
            **(self.chunk_setting),
        )

        results = [[] for _ in range(len(documents))]
        exceptions = [False for _ in range(len(documents))]
        for start in range(0, len(documents), _NORMAL_QA_MAX_WORKER):
            end = min(start + _NORMAL_QA_MAX_WORKER, len(documents))
            batch = documents[start:end]

            async with asyncio.TaskGroup() as tg:
                for idx, item in enumerate(batch):
                    real_index = start + idx
                    
                    async def task(index=real_index, document=item):
                        try:

                            tmp_page_content = _IMG_RE.sub("", document.page_content)
                            logger.debug(f'{self.task_id} Idx-{index} PageContent: {repr(tmp_page_content)}')
                            result = await agenerate_qa(self.model_instance, self.language, tmp_page_content)
                            qas = result.get('data').get('qa')
                            logger.debug(f'{self.task_id} Idx-{index} QAs: {qas}')
                            for qa in qas:
                                doc = Document(page_content=qa.get('question'), metadata=copy.deepcopy(document.metadata))
                                doc.metadata['content_id'] = get_text_id(doc.page_content)
                                doc.metadata['content_hash'] = get_text_hash(doc.page_content)
                                doc.metadata['content_len'] = len_without_link(doc.page_content)
                                doc.metadata['answer'] = qa.get('answer')
                                doc.metadata['answer_len'] = len_without_link(qa.get('answer'))
                                results[index].append(doc)
                        except Exception as e:
                            logger.warning(f'{self.task_id} Idx-{index} failed for:\n{traceback.format_exc()}')
                            exceptions[index] = True
                            pass

                    tg.create_task(task())

        if all(exceptions):
            raise RuntimeError('All chunks failed All chunks failed when generate q&a.')

        new_documents = list(itertools.chain.from_iterable(results))

        return new_documents
