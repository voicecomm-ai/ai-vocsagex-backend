from typing import List, Dict, Tuple, Optional

from core.rag.reranker.reranker_base import BaseReranker
from core.rag.reranker.reranker_model import ModelReranker
from core.rag.reranker.reranker_weight import WeightReranker
from core.rag.entities.document import Document

_reranker_map = {
    'MODEL': ModelReranker,
    'WEIGHT': WeightReranker,
}

class RerankProcessor:

    @classmethod
    def get_reranker(cls, reranker_type: str, **kwargs) -> BaseReranker:
        if reranker_type not in _reranker_map.keys():
            raise ValueError(f'Unsupported reranker type: {reranker_type}')
        return _reranker_map[reranker_type](**kwargs)
    
    @classmethod
    async def rerank(
        cls, 
        reranker: BaseReranker,
        query: str, 
        documents: List[Document], 
        top_k: int, 
        score_threshold: Optional[float] = None
    ) -> Tuple[List[Document], Dict]:
        rerank_documents, usage = await reranker.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
            score_threshold=score_threshold
        )

        return rerank_documents, usage