from typing import Optional, Any, List
import copy

from core.rag.retriever.retrieve_processor import (
    RetrieveProcessor, 
    RetrieveConfig, 
    RecallConfig, 
)
from core.rag.utils.rag_utils import add_usage_dict

async def aknowledge_retrieve(
    query: str, 
    knowledge_base_list: List[RetrieveConfig],
    is_recall: bool,
    knowledge_recall_config: Optional[RecallConfig],
    is_metadata_filter: bool = False,
    metadata_mode: Optional[str] = None,
    metadata_info: Optional[Any] = None,
):
    output = {
        'done': True,
        'data': {
            'result': []
        },
        'usage': None
    }

    documents_list = []
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for retrieve_config in knowledge_base_list:
        retrieve_documents, retrieve_usage = await RetrieveProcessor.retrieve(
            query=query,
            retrieve_config=retrieve_config,
            is_metadata_filter=is_metadata_filter,
            metadata_mode=metadata_mode,
            metadata_info=metadata_info,
        )

        # 将所有的document的向量置None，再召回时重新生成，避免后续处理时，维度不一
        for doc in retrieve_documents:
            doc.vector = None

        documents_list.append(retrieve_documents)
        usage = add_usage_dict(usage, retrieve_usage)

    if is_recall:
        recall_documents, recall_usage = await RetrieveProcessor.recall(
            query=query, 
            documents_list=documents_list,
            recall_config=knowledge_recall_config,
        )
        usage = add_usage_dict(usage, recall_usage)
    else:
        recall_documents = [doc for sublist in documents_list for doc in sublist]

    for document in recall_documents:
        metadata = copy.deepcopy(document.metadata)
        metadata.pop('context_content')

        item = {
            "content": document.metadata.get('context_content', ''),
            "title": document.metadata.get('title', ''),
            "url": metadata.get('source', ''),
            "icon": "",
            "metadata": metadata,
        }
        output['data']['result'].append(item)
    output['usage'] = usage

    return output