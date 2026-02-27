from typing import List, Optional, Dict, Tuple

from core.model.model_manager import ModelManager, ModelInstanceType
from core.rag.reranker.reranker_base import BaseReranker
from core.rag.entities.document import Document

class ModelReranker(BaseReranker):
    def __init__(self, model_instance_provider: str, model_instance_config: Dict, **kwargs):
        '''
            此处使用的是Rerank模型
        '''
        self.model_rerank = ModelManager.get_model_instance(
            model_instance_provider,
            ModelInstanceType.Rerank,
            **model_instance_config
        )

    async def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int, 
        score_threshold: Optional[float] = None
    ) -> Tuple[List[Document], Dict]:
        # documents 去重
        unique_documents = []
        content_id_set = set()
        for document in documents:
            if document.metadata is not None and 'content_id' in document.metadata.keys() and document.metadata['content_id'] not in content_id_set:
                content_id_set.add(document.metadata['content_id'])
                unique_documents.append(document)
        documents = unique_documents

        rerank_results = await self.model_rerank.ainvoke_rerank(
            query=query,
            documents=[document.page_content for document in documents],
            top_k=top_k,
            return_documents=False
        )

        rerank_documents = []
        for result in rerank_results.results:
            idx = result.index
            score = result.score

            if score_threshold and score < score_threshold:
                continue

            if documents[idx].metadata is not None:
                documents[idx].metadata['score'] = score
                rerank_documents.append(documents[idx])
        
        return rerank_documents, rerank_results.usage.to_dict()