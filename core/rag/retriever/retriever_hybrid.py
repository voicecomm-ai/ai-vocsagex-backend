from typing import List, Optional, Dict, Tuple, Any

from nltk.corpus.reader import documents
from pydantic import BaseModel, Field

from config.config import get_config
from core.database.database_factory import DatabaseFactory
from core.rag.entities.document import Document
from core.rag.retriever.retriever_base import BaseRetriever
from core.rag.utils.sql_operation import select_vector_by_knowledge_base_id, \
    full_text_search_by_knowledge_base_id
from core.rag.vectorizer.vectorize_processor import VectorizeProcessor
from core.rag.reranker.rerank_processor import RerankProcessor
from core.rag.utils.rag_utils import add_usage_dict, list_to_pgvector_str
from core.rag.metadata.metada_processor import MetadataProcessor

class HybridRetrieverConfig(BaseModel):
    embedding_model_instance_provider: str = Field(..., description='Embedding模型供应商')
    embedding_model_instance_config: Dict = Field(..., description='Embedding模型配置信息')
    hybrid_rerank_type: str = Field(..., description='重排序策略')
    hybrid_semantic_weight: Optional[float] = Field(None, description='权重设置：语义权重')
    rerank_model_instance_provider: Optional[str] = Field(None, description='Rerank模型供应商')
    rerank_model_instance_config: Optional[Dict] = Field(None, description='Rerank模型配置信息')

    class Config:
        extra = 'allow'

class HybridRetriever(BaseRetriever):
    def __init__(self, **kwargs):
        self.retriever_config_raw = kwargs
        self.retriever_config = HybridRetrieverConfig.model_validate(kwargs)
        self.need_vector = False

        # 初始化向量生成器
        self.vectorizer = VectorizeProcessor.get_vectorizer(
            'normal',
            model_instance_provider=self.retriever_config.embedding_model_instance_provider, 
            model_instance_config=self.retriever_config.embedding_model_instance_config, 
        )

        # 初始化重排序
        if self.retriever_config.hybrid_rerank_type == 'MODEL':
            self.reranker = RerankProcessor.get_reranker(
                'MODEL',
                model_instance_provider=self.retriever_config.rerank_model_instance_provider,
                model_instance_config=self.retriever_config.rerank_model_instance_config,
            )
            self.need_vector = False
        elif self.retriever_config.hybrid_rerank_type == 'WEIGHT':
            self.reranker = RerankProcessor.get_reranker(
                'WEIGHT',
                weight=self.retriever_config.hybrid_semantic_weight,
                model_instance_provider=self.retriever_config.embedding_model_instance_provider, 
                model_instance_config=self.retriever_config.embedding_model_instance_config, 
            )
            self.need_vector = True
        else:
            raise ValueError(f'Unsupported rerank type: {self.retriever_config.hybrid_rerank_type}')

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
        # 生成query向量
        vectorize_result = await VectorizeProcessor.vectorize(self.vectorizer, [query])
        query_vector = vectorize_result[0]['vector']
        usage = vectorize_result[0]['usage']

        # 获取数据库实例
        database_info = get_config()["dependent_info"]["database"]
        db = DatabaseFactory.get_database(database_info["type"])

        metadata_condition = None
        if is_metadata_filter:
            metadata_condition, metadata_usage = await MetadataProcessor.transform_metadata_condition(
                query=query,
                metadata_mode=metadata_mode,
                metadata_info=metadata_info,
            )

            usage = add_usage_dict(usage, metadata_usage)

        # 通过query从表中查询top_k条记录
        documents = await full_text_search_by_knowledge_base_id(
            db, self.need_vector, knowledge_base_id, top_k, query, score_threshold
        )



        # 通过向量从表中查询top_k，并获取score
        vector_documents = await select_vector_by_knowledge_base_id(
            db, 
            self.need_vector, 
            knowledge_base_id, 
            top_k, 
            list_to_pgvector_str(query_vector), 
            score_threshold,
            metadata_condition,
        )

        # 根据rerank类型进行rerank
        documents.extend(vector_documents)
        if documents:
            documents, rerank_usage = await RerankProcessor.rerank(
                reranker=self.reranker,
                query=query,
                documents=documents,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            usage = add_usage_dict(usage, rerank_usage)

        return documents, usage
