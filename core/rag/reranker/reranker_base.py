from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict

from core.rag.entities.document import Document


class BaseReranker(ABC):

    @abstractmethod
    async def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int, 
        score_threshold: Optional[float] = None
    ) -> Tuple[List[Document], Dict]:
        raise NotImplementedError