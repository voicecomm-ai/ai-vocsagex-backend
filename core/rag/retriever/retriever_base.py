from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any

from core.rag.entities.document import Document
from core.rag.metadata.metada_processor import MetadataMode

TOP_K_MAX = 30

class BaseRetriever(ABC):

    @abstractmethod
    async def retrieve(
        self, 
        query: str, 
        knowledge_base_id: int, 
        top_k: int, 
        score_threshold: Optional[float] = None, 
        is_metadata_filter: bool = False,
        metadata_mode: Optional[MetadataMode] = None,
        metadata_info: Optional[Any] = None,
        **kwargs
    ) -> Tuple[List[Document], Dict]:
        raise NotImplementedError