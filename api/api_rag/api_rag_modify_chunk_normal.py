'''
    编辑分段 - 普通分段

    根据段落内容，生成向量
    修改表knowledge_base_document中对应记录的preview_chunks、word_count列
    修改表knowledge_base_doc_vector中的原记录的retrieve_content、context_content、metadata、usage、vector列
'''

from typing import Dict
import traceback
import json

from pydantic import BaseModel, Field, model_validator
from fastapi import Request, Body

from api.api import app, get_api_client_tag
from core.database.database_factory import DatabaseFactory
from core.rag.vectorizer.vectorize_processor import VectorizeProcessor
from core.rag.utils.sql_expression import (
    SQL_EXPRESSION_DOCUMENT_UPDATE_CURD, 
    SQL_EXPRESSION_VECTOR_UPDTAE_CURD, 
    SQL_EXPRESSION_DOCUMENT_SELECT_CURD, 
    SQL_EXPRESSION_VECTOR_SELECT_CURD, 
)
from core.rag.utils.rag_utils import (
    len_without_link, 
    list_to_pgvector_str, 
    get_text_hash, 
)
from config.config import get_config
from logger import get_logger

logger = get_logger('api')

class ApiRagRequestModel(BaseModel):
    document_id: int = Field(..., description='文档id')
    chunk_id: int = Field(..., description='段落id')
    chunk_content: str = Field(..., description='段落内容')
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
    path='/Voicecomm/VoiceSageX/Rag/ModifyChunkNormal', 
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

        content_hash = get_text_hash(request.chunk_content)
        new_character = len_without_link(request.chunk_content)

        logger.debug(f'{task_id} Vectorized.')
        logger.debug(f'{task_id} new character:{new_character}')

    except Exception as e:
        logger.error(f'{task_id} Fail to process data for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to process data for:{type(e).__name__}: {str(e)}.'
        )

    # 事务
    try:
        async with db.conn() as conn:
            async with conn.transaction():
                # 获取knowledge_base_document中的原记录
                document_select_sql = SQL_EXPRESSION_DOCUMENT_SELECT_CURD.format(
                    ','.join(['preview_chunks', 'word_count', ])
                )
                rows = await conn.fetch(document_select_sql, request.document_id)
                if not rows or len(rows) > 1:
                    raise ValueError(f"No records or not unique.")
                record = dict(rows[0])
                record['preview_chunks'] = json.loads(record['preview_chunks'])
                logger.debug(f'{task_id} Select record from "knowledge_base_document".')

                # 获取knowledge_base_doc_vector中chunk的metedata
                chunk_select_sql = SQL_EXPRESSION_VECTOR_SELECT_CURD.format(
                    ','.join(['metadata'])
                )
                rows = await conn.fetch(chunk_select_sql, request.chunk_id)
                if not rows or len(rows) > 1:
                    raise ValueError(f"No records or not unique.")
                chunk_record = dict(rows[0])
                chunk_record['metadata'] = json.loads(chunk_record['metadata'])
                logger.debug(f'{task_id} Select record from "knowledge_base_doc_vector".')
                logger.debug(f'{task_id} chunk metadata:\n{json.dumps(chunk_record["metadata"], ensure_ascii=False, indent=4)}')

                # 在preview_chunks中找到对应的片段，并修改
                target_chunk = None
                for chunk in record['preview_chunks']:
                    if chunk['primary_key'] == request.chunk_id:
                        target_chunk = chunk
                        logger.debug(f'{task_id} Original:\n{json.dumps(target_chunk, ensure_ascii=False, indent=4)}')
                        break

                if not target_chunk:
                    raise ValueError("chunk not found.")
                
                logger.debug(f'{task_id} Locate to the specified chunk, primary key [{request.chunk_id}].')
                
                # 长度 - 原长度 + 新长度
                logger.debug(f'{task_id} original word count: {record["word_count"]}.')
                logger.debug(f'{task_id} original chunk character: {target_chunk["character"]}.')

                record['word_count'] -= target_chunk['character']
                record['word_count'] += new_character

                logger.debug(f'{task_id} new word count: {record["word_count"]}.')

                # 在preview_chunks中修改指定段落的content, character
                target_chunk['content'] = request.chunk_content
                target_chunk['character'] = new_character
                target_chunk['isEdited'] = True

                # 更新metadata
                metadata = chunk_record['metadata']
                metadata['content_hash'] = content_hash
                metadata['content_len'] = new_character
                logger.debug(f'{task_id} new metadata:\n{json.dumps(metadata, ensure_ascii=False, indent=4)}')

                # 修改表knowledge_base_document中对应记录的preview_chunks、word_count列
                update_document_sql = SQL_EXPRESSION_DOCUMENT_UPDATE_CURD.format(
                    ','.join(["preview_chunks=$2::jsonb", "word_count=$3"])
                )
                await conn.execute(
                    update_document_sql, 
                    request.document_id, 
                    json.dumps(record['preview_chunks']), 
                    record['word_count'], 
                )
                logger.debug(f'{task_id} Update "knowledge_base_document".')

                # 修改表knowledge_base_doc_vector中的原记录的retrieve_content、context_content、metadata、usage、vector列
                update_chunk_sql = SQL_EXPRESSION_VECTOR_UPDTAE_CURD.format(
                    ','.join([
                        "retrieve_content=$2",
                        "context_content=$3", 
                        "metadata=$4::jsonb",
                        "usage=$5::jsonb",
                        "vector=$6::vector",
                        "content_hash=$7",
                    ])
                )
                await conn.execute(
                    update_chunk_sql, 
                    [request.chunk_id], 
                    request.chunk_content, 
                    request.chunk_content, 
                    json.dumps(metadata), 
                    json.dumps(usage), 
                    list_to_pgvector_str(vector), 
                    content_hash, 
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