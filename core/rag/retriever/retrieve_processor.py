from typing import Dict, Tuple, List, Optional, Any, Literal
import traceback
import json

from pydantic import BaseModel, Field
from langchain.tools import tool, BaseTool
from langchain_core.tools import ToolException

from core.rag.retriever.retriever_base import BaseRetriever
from core.rag.retriever.retriever_vector import VectorRetriever
from core.rag.retriever.retriever_fulltext import FulltextRetriever
from core.rag.retriever.retriever_hybrid import HybridRetriever
from core.rag.reranker.rerank_processor import RerankProcessor
from core.rag.entities.document import Document
from core.rag.utils.rag_utils import add_usage_dict, short_unique_id
from logger import get_logger

logger = get_logger('rag')

_retriever_map = {
    'VECTOR': VectorRetriever,
    'FULL_TEXT': FulltextRetriever,
    'HYBRID': HybridRetriever,
}

class RetrieveConfig(BaseModel):
    '''
        检索配置
    '''
    knowledge_base_id: int = Field(..., description='知识库id')
    knowledge_base_description: Optional[str] = Field(None, description='知识库描述')
    knowledge_base_retrieve_type: str = Field(..., description='知识库检索类型')
    knowledge_base_retrieve_config: Dict = Field(..., description='知识库检索配置')

class RecallConfig(BaseModel):
    '''
        召回配置
    '''
    rerank_type: Literal['MODEL', "WEIGHT"] = Field(..., description='重排序策略')
    top_k: int = Field(..., description='top_k')
    score_threshold: Optional[float] = Field(None, description='score阈值')
    semantic_weight: Optional[float] = Field(None, description='权重设置：语义权重')
    embedding_model_instance_provider: Optional[str] = Field(None, description='Embedding模型供应商')
    embedding_model_instance_config: Optional[Dict] = Field(None, description='Embedding模型配置信息')
    rerank_model_instance_provider: Optional[str] = Field(None, description='Rerank模型供应商')
    rerank_model_instance_config: Optional[Dict] = Field(None, description='Rerank模型配置信息')

class RetrieveProcessor:
    
    @staticmethod
    def make_async_tool(
        retrieve_config: RetrieveConfig, 
        is_metadata_filter: bool = False,
        metadata_mode: Optional[str] = None,
        metadata_info: Optional[Any] = None,
    ) -> BaseTool:
        tool_name = f'knowledge_base_retrieve_{retrieve_config.knowledge_base_id}'
        if retrieve_config.knowledge_base_description:
            tool_description = (
                'Retrieves relevant information from the knowledge base '
                f'described as: {retrieve_config.knowledge_base_description}. '
                'When the user query is related to the entities, topics, or keywords of this knowledge base '
                '(e.g., park names, location advantages, supporting facilities, transportation, policies), '
                'you MUST call this tool first using the original user query as `query`. '
                '\n\n'
                '**IMPORTANT**: After retrieving information, you MUST evaluate its relevance to the user query:\n'
                '- If the retrieved content is relevant and helpful, use it to answer.\n'
                '- If the retrieved content is irrelevant or unrelated to the query, explicitly state that '
                'the knowledge base does not contain relevant information, and then answer using your own knowledge.\n'
                '- NEVER force-fit irrelevant retrieved content into your answer just because it was returned.'
            )
        else:
            tool_description = (
                'Retrieves relevant information from a broad, general-purpose knowledge base. '
                'If the user query mentions entities or topics likely covered by this knowledge base, '
                'you MUST call this tool first using the original user query as `query`. '
                '\n\n'
                '**IMPORTANT**: After retrieving information, you MUST evaluate its relevance to the user query:\n'
                '- If the retrieved content is relevant and helpful, use it to answer.\n'
                '- If the retrieved content is irrelevant or unrelated to the query, explicitly state that '
                'the knowledge base does not contain relevant information, and then answer using your own knowledge.\n'
                '- NEVER force-fit irrelevant retrieved content into your answer just because it was returned.'
            )

        class ToolInput(BaseModel):
            query: str = Field(..., description='Query for the knowledge base to be used to retrieve the knowledge base.')

        @tool(name_or_callable=tool_name, description=tool_description, args_schema=ToolInput)
        async def knowledge_base_retrieve(query: str) -> str:
            try:
                documents, usage = await RetrieveProcessor.retrieve(
                    query=query,
                    retrieve_config=retrieve_config,
                    is_metadata_filter=is_metadata_filter,
                    metadata_mode=metadata_mode,
                    metadata_info=metadata_info,
                )
                # 打印结构化命中文档信息
                try:
                    debug_docs = [
                        {
                            'document_id': doc.metadata.get('document_id'),
                            'score': doc.metadata.get('score'),
                            'content_length': len(doc.metadata.get('context_content', '') or ''),
                            'preview': (doc.metadata.get('context_content', '') or '')[:200]
                        }
                        for doc in documents
                    ]
                    logger.debug(f"[KB Debug] Retrieved {len(documents)} documents: {json.dumps(debug_docs, ensure_ascii=False)}")
                except Exception:
                    logger.debug("[KB Debug] Failed to build debug docs payload.")
                res = ' '.join([document.metadata.get('context_content', '') for document in documents])
                return res if res else 'No relevant information found in the knowledge base for this query. Please use your own knowledge to answer the user\'s question.'
            except Exception as e:
                logger.error(f'Failed to call [{tool_name}] for {traceback.format_exc()}')
                raise ToolException(f'{type(e).__name__}: {str(e)}')

        return knowledge_base_retrieve

    @classmethod
    async def retrieve(
        cls, 
        query: str, 
        retrieve_config: RetrieveConfig,
        is_metadata_filter: bool = False,
        metadata_mode: Optional[str] = None,
        metadata_info: Optional[Any] = None,
    ) -> Tuple[List[Document], Dict]:
    
        retriever_type = retrieve_config.knowledge_base_retrieve_type
        retriever_config = retrieve_config.knowledge_base_retrieve_config
        retriever: BaseRetriever = _retriever_map[retriever_type](**retriever_config)
        
        documents, usage = await retriever.retrieve(
            query=query,
            knowledge_base_id=retrieve_config.knowledge_base_id,
            top_k=retrieve_config.knowledge_base_retrieve_config.get('top_k'),
            score_threshold=retrieve_config.knowledge_base_retrieve_config.get('score_threshold'),
            is_metadata_filter=is_metadata_filter,
            metadata_mode=metadata_mode,
            metadata_info=metadata_info,
        )

        return documents, usage

    @classmethod
    async def recall(
        cls,
        query: str,
        documents_list: List[List[Document]],
        recall_config: Optional[RecallConfig] = None,
    ) -> Tuple[List[Document], Dict]:
        documents = [doc for sublist in documents_list for doc in sublist]

        if recall_config.rerank_type == 'MODEL':
            reranker = RerankProcessor.get_reranker(
                reranker_type='MODEL',
                model_instance_provider=recall_config.rerank_model_instance_provider,
                model_instance_config=recall_config.rerank_model_instance_config,
            )
        elif recall_config.rerank_type == 'WEIGHT':
            reranker = RerankProcessor.get_reranker(
                reranker_type='WEIGHT',
                weight=recall_config.semantic_weight,
                model_instance_provider=recall_config.embedding_model_instance_provider,
                model_instance_config=recall_config.embedding_model_instance_config,
            )
        else:
            raise ValueError(f'Unsupported rerank type: {recall_config.rerank_type}')

        documents, usage = await RerankProcessor.rerank(
            reranker=reranker,
            query=query,
            documents=documents,
            top_k=recall_config.top_k,
            score_threshold=recall_config.score_threshold,
        )

        return documents, usage
        