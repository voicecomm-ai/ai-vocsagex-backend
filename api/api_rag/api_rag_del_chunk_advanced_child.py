'''
    删除分段 - 高级分段 - 子分段

    修改表knowledge_base_document中对应记录的preview_chunks、word_count列
    删除表knowledge_base_doc_vector中的记录
    更新表knowledge_base_doc_vector中当前父段下，所有子段的context_content
'''

from typing import Dict
import traceback
import json

from pydantic import BaseModel, Field
from fastapi import Request, Body

from api.api import app, get_api_client_tag
from core.database.database_factory import DatabaseFactory
from core.rag.utils.sql_expression import (
    SQL_EXPRESSION_DOCUMENT_UPDATE_CURD, 
    SQL_EXPRESSION_VECTOR_DELETE_CURD, 
    SQL_EXPRESSION_VECTOR_UPDTAE_CURD, 
    SQL_EXPRESSION_DOCUMENT_SELECT_CURD, 
)
from config.config import get_config
from logger import get_logger

logger = get_logger('api')

class ApiRagRequestModel(BaseModel):
    document_id: int = Field(..., description='文档id')
    parent_idx: int = Field(..., description='父段序号')
    child_chunk_id: int = Field(..., description='子段段落id')

class ApiRagResponseModel(BaseModel):
    code: int = Field(..., description='状态码，成功为1000，失败为2000')
    msg: str = Field(..., description='状态信息')

@app.post(
    path='/Voicecomm/VoiceSageX/Rag/DelChunkAdvancedChild', 
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
        logger.debug(f'{task_id} Got required instance.')
    except Exception as e:
        logger.error(f'{task_id} Fail to get required instance for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to get required instance for:{type(e).__name__}: {str(e)}.'
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

                # 在preview_chunks中找到对应的父段，在父段下找到对应子段，并删除
                target_parent_chunk = None
                target_child_chunk = None
                for chunk in record['preview_chunks']:
                    if chunk['id'] == request.parent_idx:
                        target_parent_chunk = chunk
                        for child_chunk in chunk['content']:
                            if child_chunk['primary_key'] == request.child_chunk_id:
                                target_child_chunk = child_chunk
                                # 删除
                                logger.debug(f'{task_id} Delete:\n{json.dumps(target_child_chunk, ensure_ascii=False, indent=4)}')
                                chunk['content'].remove(child_chunk)
                                # 更新父段长度
                                logger.debug(f'{task_id} original parent chunk character: {chunk["character"]}.')
                                chunk['character'] -= target_child_chunk['character']
                                logger.debug(f'{task_id} original parent chunk character: {chunk["character"]}.')
                                chunk['isEdited'] = True
                                break
                        break

                if not target_child_chunk:
                    raise ValueError("chunk not found.")

                logger.debug(f'{task_id} Locate to the specified parent chunk, index [{request.parent_idx}].')
                logger.debug(f'{task_id} Locate to the specified child chunk, primary key [{request.child_chunk_id}].')

                # 记录当前父段的所有子段主键，并组装所有子段的context_content
                context_content = []
                child_pks = []
                for child_chunk in target_parent_chunk['content']:
                    child_pks.append(child_chunk['primary_key'])
                    context_content.append(child_chunk['content'])
                
                context_content = ''.join(context_content)
                logger.debug(f'{task_id} primary keys need to be modified:\n{child_pks}.')
                logger.debug(f'{task_id} new context_content: {context_content}')

                # 长度减去删除的片段长度
                logger.debug(f'{task_id} original word count: {record["word_count"]}.')

                record['word_count'] -= target_child_chunk['character']

                logger.debug(f'{task_id} new word count: {record["word_count"]}.')

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

                # 删除表knowledge_base_doc_vector中的记录
                await conn.execute(
                    SQL_EXPRESSION_VECTOR_DELETE_CURD,
                    [request.child_chunk_id],
                )
                logger.debug(f'{task_id} Delete record from "knowledge_base_doc_vector".')
                
                # 更新表knowledge_base_doc_vector中当前父段下，所有子段的context_content
                update_chunk_sql = SQL_EXPRESSION_VECTOR_UPDTAE_CURD.format(
                    ','.join(["context_content=$2"])
                )
                await conn.execute(
                    update_chunk_sql,
                    child_pks,
                    context_content,
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