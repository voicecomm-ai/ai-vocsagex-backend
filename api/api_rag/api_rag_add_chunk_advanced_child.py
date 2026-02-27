'''
    添加分段 - 高级分段 - 子分段

    根据段落内容，生成向量
    增加记录至表knowledge_base_doc_vector
    修改表knowledge_base_document中对应记录的preview_chunks、word_count列
    修改表knowledge_base_doc_vector中，当前父段下所有子段的retrieve_content列
'''

from typing import Dict, Literal
import traceback
import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from fastapi import Request, Body

from api.api import app, get_api_client_tag
from core.database.database_factory import DatabaseFactory
from core.rag.vectorizer.vectorize_processor import VectorizeProcessor
from core.rag.utils.sql_expression import (
    SQL_EXPRESSION_DOCUMENT_SELECT_CURD, 
    SQL_EXPRESSION_DOCUMENT_UPDATE_CURD, 
    SQL_EXPRESSION_VECTOR_INSERT_CURD, 
    SQL_EXPRESSION_VECTOR_UPDTAE_CURD, 
)
from core.rag.utils.rag_utils import (
    len_without_link, 
    list_to_pgvector_str, 
    get_text_hash, 
    get_text_id, 
    generate_preview_chunks_from_documents, 
)
from core.rag.splitter.splitter_entities import SplitType
from api.api_rag.entities import ChunkStatus
from core.rag.entities.document import Document
from config.config import get_config
from logger import get_logger

logger = get_logger('api')

class ApiRagRequestModel(BaseModel):
    knowledge_base_id: int = Field(..., description='知识库id')
    document_id: int = Field(..., description='文档id')
    parent_idx: int = Field(..., description='父段序号')
    chunk_content: str = Field(..., description='段落内容')
    chunk_status: Literal['ENABLE', 'DISABLE'] = Field(..., description='分段是否启用')
    model_instance_provider: str = Field(..., description='Embedding模型加载方式')
    model_instance_config: Dict = Field(..., description='Embedding模型配置信息')

    @model_validator(mode='after')
    def validate_input(self) -> 'ApiRagRequestModel':
        if not self.chunk_content:
            raise ValueError('Illegal chunk_content.')
        return self
    
class ApiRagResponseModel(BaseModel):
    code: int = Field(..., description='状态码，成功为1000，失败为2000')
    msg: str = Field(..., description='状态信息')

@app.post(
    path='/Voicecomm/VoiceSageX/Rag/AddChunkAdvancedChild', 
    response_model=ApiRagResponseModel,
    response_model_exclude_none=False, 
)
async def handler(conn: Request, body: Dict = Body(...)):
    tag, task_id = get_api_client_tag(conn)
    logger.debug(f'{task_id} {tag}')

    # 格式校验
    try:
        logger.debug(f'{task_id} Request body:\n{json.dumps(body, indent=4, ensure_ascii=False)}')
        request = ApiRagRequestModel.model_validate(body)
    except Exception as e:
        logger.error(f'{task_id} Fail to validate pydantic instance for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: The request body field is incorrect for:{type(e).__name__}: {str(e)}.'
        )
    
    # 获取数据库实例
    try:
        database_info = get_config()['dependent_info']['database']
        db = DatabaseFactory.get_database(database_info['type'])

        document_path_prefix = get_config().get('dependent_info').get('knowledge_base').get('document_path_prefix')

        vectorizer = VectorizeProcessor().get_vectorizer(
            model_instance_provider=request.model_instance_provider, 
            model_instance_config=request.model_instance_config, 
        )

        logger.debug(f'{task_id} Got required instance.')
    except Exception as e:
        logger.error(f'{task_id} Fail to get required instance for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to get required instance for:{type(e).__name__}: {str(e)}.'
        )
    
    # 处理数据
    try:
        # 生成向量
        outputs = await VectorizeProcessor.vectorize(vectorizer, [request.chunk_content])
        vector = outputs[0]['vector']
        usage = outputs[0]['usage']

        content_id = get_text_id(request.chunk_content)
        content_len = len_without_link(request.chunk_content)
        content_hash = get_text_hash(request.chunk_content)

        new_doc = Document(
            page_content=request.chunk_content,
            vector=vector,
            metadata={
                "source": "Additional chunk",
                "content_id": content_id,
                "content_len": content_len,
                "content_hash": content_hash,
            },
        )
        logger.debug(f'{task_id} Vectorized.')
        logger.debug(f'{task_id} content_id: {content_id}')
        logger.debug(f'{task_id} content_len: {content_len}')

    except Exception as e:
        logger.error(f'{task_id} Fail to process data for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to process data for:{type(e).__name__}: {str(e)}.'
        )

    # 事务
    try:
        # 从表 knowledge_base_document 中读取文档的记录
        # 生成 chunk_record
        # 增加记录至表knowledge_base_doc_vector
        # 修改表knowledge_base_document中对应记录的preview_chunks、word_count列
        # 修改表knowledge_base_doc_vector中，当前父段下所有子段的retrieve_content列

        async with db.conn() as conn:
            async with conn.transaction():
                # 从表 knowledge_base_document 中读取文档的记录
                document_select_sql = SQL_EXPRESSION_DOCUMENT_SELECT_CURD.format(
                    ','.join(['preview_chunks', 'word_count', 'chunking_strategy', 'name', 'unique_name'])
                )
                rows = await conn.fetch(document_select_sql, request.document_id)
                if not rows or len(rows) > 1:
                    raise ValueError(f"No records or not unique.")
                record = dict(rows[0])
                record['preview_chunks'] = json.loads(record['preview_chunks'])
                logger.debug(f'{task_id} Select record from "knowledge_base_document".')
                logger.debug(f'{task_id} chunking_strategy: {record["chunking_strategy"]}')
                logger.debug(f'{task_id} name: {record["name"]}')
                logger.debug(f'{task_id} unique_name: {record["unique_name"]}')

                # 根据parent_idx，定位到父段
                target_chunk = None
                for chunk in record['preview_chunks']:
                    if chunk['id'] == request.parent_idx:
                        target_chunk = chunk
                        break

                if not target_chunk:
                    raise ValueError("chunk not found.")
                
                logger.debug(f'{task_id} Locate to the specified parent chunk, index [{request.parent_idx}].')

                # 生成子段序号及其对应的父段序号
                if target_chunk['content']:
                    new_doc.metadata["idx"] = target_chunk['content'][-1]["id"] + 1
                    new_doc.metadata["f_idx"] = target_chunk['id']
                    new_doc.metadata['source'] = str(Path(document_path_prefix) / 'knowledge-base/documents/' / record['unique_name'])
                    new_doc.metadata['title'] = record['name']
                    logger.debug(f'{task_id} new child chunk index: {new_doc.metadata["idx"]}.')

                # 生成原始 context_content
                ori_context_content = ''.join(c['content'] for c in target_chunk["content"])
                new_context_content = ori_context_content + request.chunk_content
                logger.debug(f'{task_id} new context content: {new_context_content}')

                # 生成原始 子段ids
                child_pks = [c['primary_key'] for c in target_chunk["content"]]

                # 生成 chunk_record
                chunk_record = (
                    content_id, 
                    request.knowledge_base_id,
                    request.document_id,
                    request.chunk_content,
                    new_context_content,
                    json.dumps(new_doc.metadata),
                    list_to_pgvector_str(vector),
                    json.dumps(usage),
                    ChunkStatus.SUCCESS.value,
                    content_hash,
                    record['chunking_strategy'],
                    request.chunk_status, 
                )

                # 增加记录至表knowledge_base_doc_vector
                id = await conn.fetchrow(SQL_EXPRESSION_VECTOR_INSERT_CURD, *chunk_record)
                logger.debug(f'{task_id} inserted chunk record: {id["id"]}.')

                # 更新metadata
                new_doc.metadata['primary_key'] = id['id']

                # 生成新增chunk的preview_chunks
                new_preview_chunks = generate_preview_chunks_from_documents(
                    SplitType.NORMAL.value,
                    [new_doc],
                    is_edited=True,
                    chunk_status=request.chunk_status,
                )
                logger.debug(f'{task_id} additional preview chunks:\n{json.dumps(new_preview_chunks, ensure_ascii=False, indent=4)}')

                child_pks.append(new_doc.metadata['primary_key'])
                logger.debug(f'{task_id} primary keys need to be modified:\n{child_pks}.')

                # 修改表knowledge_base_document中对应记录的preview_chunks、word_count列
                logger.debug(f'{task_id} original parent chunk character: {target_chunk["character"]}.')
                logger.debug(f'{task_id} original word count: {record["word_count"]}.')

                target_chunk['content'].extend(new_preview_chunks)
                target_chunk['character'] += new_doc.metadata['content_len']
                target_chunk['isEdited'] = True
                record['word_count'] += new_doc.metadata['content_len']

                logger.debug(f'{task_id} new parent chunk character: {target_chunk["character"]}.')
                logger.debug(f'{task_id} new word count: {record["word_count"]}.')

                # 更新表 knowledge_base_document
                document_update_sql = SQL_EXPRESSION_DOCUMENT_UPDATE_CURD.format(
                    ','.join(["preview_chunks=$2::jsonb", "word_count=$3"])
                )
                await conn.execute(
                    document_update_sql, 
                    request.document_id, 
                    json.dumps(record['preview_chunks']),
                    record['word_count'],
                )
                logger.debug(f'{task_id} Update "knowledge_base_document".')

                # 修改表knowledge_base_doc_vector中，当前父段下所有子段的context_content列
                chunk_update_sql = SQL_EXPRESSION_VECTOR_UPDTAE_CURD.format(
                    ','.join(["context_content=$2"])
                )
                await conn.execute(
                    chunk_update_sql,
                    [child_pks],
                    new_context_content,
                )
                logger.debug(f'{task_id} Update "knowledge_base_doc_vector".')

    except Exception as e:
        logger.error(f'{task_id} Fail to excute sql transaction for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to excute sql transaction for:{type(e).__name__}: {str(e)}.'
        )
    
    # 响应
    response = ApiRagResponseModel(
        code=1000, 
        msg=f'{task_id} Success.'
    )
    logger.debug(f'{task_id} Response:\n{response.model_dump_json(indent=4)}')

    return response