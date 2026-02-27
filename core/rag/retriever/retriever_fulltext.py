from typing import List, Optional, Dict, Tuple, Any

from pydantic import BaseModel, Field

from config.config import get_config
from core.database.database_factory import DatabaseFactory
from core.rag.entities.document import Document
from core.rag.retriever.retriever_base import BaseRetriever, TOP_K_MAX
from core.rag.reranker.rerank_processor import RerankProcessor
from core.rag.utils.rag_utils import add_usage_dict
from core.rag.utils.sql_operation import full_text_search_by_knowledge_base_id
from core.rag.metadata.metada_processor import MetadataProcessor


class FulltextRetrieverConfig(BaseModel):
    embedding_model_instance_provider: str = Field(..., description='Embedding模型供应商')
    embedding_model_instance_config: Dict = Field(..., description='Embedding模型配置信息')
    is_rerank: bool = Field(..., description='是否开启rerank')
    rerank_model_instance_provider: Optional[str] = Field(None, description='Rerank模型供应商')
    rerank_model_instance_config: Optional[Dict] = Field(None, description='Rerank模型配置信息')

    class Config:
        extra = 'allow'

class FulltextRetriever(BaseRetriever):
    def __init__(self, **kwargs):
        self.retriever_config_raw = kwargs
        self.retriever_config = FulltextRetrieverConfig.model_validate(kwargs)

        # 初始化重排序
        if self.retriever_config.is_rerank:
            self.reranker = RerankProcessor.get_reranker(
                'MODEL',
                model_instance_provider=self.retriever_config.rerank_model_instance_provider,
                model_instance_config=self.retriever_config.rerank_model_instance_config,
            )

    async def retrieve(
        self, 
        query: str, 
        knowledge_base_id: int, 
        top_k: int, 
        score_threshold: Optional[float] = None, 
        is_metadata_filter: bool = False,
        metadata_mode: Optional[str] = None,
        metadata_info: Optional[Any] = None,
        **kwargs
    ) -> Tuple[List[Document], Dict]:
        # 获取数据库实例
        database_info = get_config()["dependent_info"]["database"]
        db = DatabaseFactory.get_database(database_info["type"])

        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        metadata_condition = None
        metadata_usage = None
        if is_metadata_filter:
            metadata_condition, metadata_usage = await MetadataProcessor.transform_metadata_condition(
                query=query,
                metadata_mode=metadata_mode,
                metadata_info=metadata_info,
            )

            usage = add_usage_dict(usage, metadata_usage)

        # 通过query从表中查询top_k条记录
        documents = await full_text_search_by_knowledge_base_id(
            db, 
            True, 
            knowledge_base_id, 
            min(top_k**2, TOP_K_MAX) if self.retriever_config.is_rerank else top_k, 
            query, 
            None if self.retriever_config.is_rerank else score_threshold,
            metadata_condition,
        )

        # 根据是否有rerank，进行rerank
        if documents:
            if self.retriever_config.is_rerank:
                documents, rerank_usage = await RerankProcessor.rerank(
                    reranker=self.reranker,
                    query=query,
                    documents=documents,
                    top_k=top_k,
                    score_threshold=score_threshold
                )
                usage = add_usage_dict(usage, rerank_usage)

        return documents, usage
