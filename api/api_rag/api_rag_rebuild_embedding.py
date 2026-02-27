'''
    知识库设置：修改Embedding模型后的重建向量
'''

from typing import Dict, Optional
import traceback
import json
import asyncio

from pydantic import BaseModel, Field
from fastapi import Request, Body

from api.api import app, get_api_client_tag
from core.database.database_factory import DatabaseFactory
from core.rag.vectorizer.vectorize_processor import VectorizeProcessor
from core.rag.utils.sql_operation import (
    select_vector_from_db, 
    update_vector_into_db,
    modify_document_status_by_knowledge_base_id,
)
from core.rag.utils.rag_utils import list_to_pgvector_str
from core.model.model_entities import TokenUsage
from api.api_rag.entities import DocumentStatus, ChunkStatus
from config.config import get_config
from logger import get_logger

logger = get_logger('api')

class ApiRagRequestModel(BaseModel):
    knowledge_base_id: int = Field(..., description='java内部知识库id')
    model_instance_provider: str = Field(..., description='模型供应商')
    model_instance_config: Dict = Field(..., description='模型配置信息')

class ApiRagResponseModel(BaseModel):
    code: int = Field(..., description='状态码，成功为1000，失败为2000')
    msg: str = Field(..., description='状态信息')
    data: Optional[Dict] = Field(None, description='数据')
    usage: Optional[Dict] = Field(None, description='模型tokens用量')

@app.post(path='/Voicecomm/VoiceSageX/Rag/RebuildEmbedding', response_model=ApiRagResponseModel, response_model_exclude_none=False)
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
        embedding_batch_size = get_config().get('server', {}).get('embedding_batch_size', 8)
        database_info = get_config()['dependent_info']['database']
        db = DatabaseFactory.get_database(database_info['type'])
        vectorizer = VectorizeProcessor.get_vectorizer(**request.model_dump())
        logger.debug(f'{task_id} Got required instance.')
    except Exception as e:
        logger.error(f'{task_id} Fail to get required instance for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to get required instance for:{type(e).__name__}: {str(e)}.'
        )

    # 修改文档库中当前知识库id下的文档状态为IN_PROGRESS
    try:
        await modify_document_status_by_knowledge_base_id(
            db,
            request.knowledge_base_id,
            DocumentStatus.IN_PROGRESS.value,
        )
    except Exception as e:
        logger.error(f'{task_id} Fail to modify document status to {DocumentStatus.SUCCESS.value} for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to modify document status for:{type(e).__name__}: {str(e)}.'
        )

    # 取出记录
    try:
        # records [(content_id, retrieve_content)]
        records = await select_vector_from_db(db, request.knowledge_base_id)
        logger.debug(f'{task_id} Select {len(records)} from database.')
        # logger.debug(f'{task_id} Records:\n{records}')
    except Exception as e:
        logger.error(f'{task_id} Fail to select records from database for:\n{traceback.format_exc()}')

        try:
            await modify_document_status_by_knowledge_base_id(
                db,
                request.knowledge_base_id,
                DocumentStatus.FAILED.value,
            )
        except Exception as e:
            logger.error(f'{task_id} Fail to modify document status to {DocumentStatus.FAILED.value} for:\n{traceback.format_exc()}')

        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to select records from database for:{type(e).__name__}: {str(e)}.'
        )

    usage: Optional[TokenUsage] = None
    if records:
        # 重新生成索引
        failure_count = 0
        for i in range(0, len(records), embedding_batch_size):
            update_tuples = []
            batch_records = records[i:i+embedding_batch_size]
            coros = [VectorizeProcessor.vectorize(vectorizer, [record[1]]) for record in batch_records]

            outputs = await asyncio.gather(
                *coros, 
                return_exceptions=True
            )

            for idx, output in enumerate(outputs):
                if isinstance(output, Exception):
                    # 错误
                    update_tuples.append((ChunkStatus.FAILED.value, None, None, batch_records[idx][0]))
                    failure_count += 1
                    logger.warning(f'{task_id} [{i + idx}] Vectorize failed for:\n{repr(output)}')
                else:
                    # 成功
                    update_tuples.append(
                        (
                            ChunkStatus.SUCCESS.value,
                            list_to_pgvector_str(output[0]['vector']),
                            json.dumps(output[0]['usage']),
                            batch_records[idx][0],
                        )
                    )

                    new_usage = TokenUsage.model_validate(output[0]['usage'])
                    usage = usage + new_usage if usage else new_usage

                    logger.debug(f'{task_id} [{i + idx}] Vectorize Done.')

            # 写入记录
            if update_tuples:
                try:
                    await update_vector_into_db(db, update_tuples)
                    logger.debug(f'{task_id} Updated records[{i}-{i+len(batch_records)-1}] into database.')
                except Exception as e:
                    logger.error(f'{task_id} Fail to update records[{i}-{i+len(batch_records)-1}] into database for:\n{traceback.format_exc()}')

        if failure_count == len(records):
            logger.error(f'{task_id} All records failed to rebuild embedding.')

            try:
                await modify_document_status_by_knowledge_base_id(
                    db,
                    request.knowledge_base_id,
                    DocumentStatus.FAILED.value,
                )
            except Exception as e:
                logger.error(f'{task_id} Fail to modify document status to {DocumentStatus.FAILED.value} for:\n{traceback.format_exc()}')

            return ApiRagResponseModel(
                code=2000,
                msg=f'{task_id} All records failed to rebuild embedding.'
            )
    else:
        logger.warning(f'{task_id} Cannot find any records in knowledge_base by id[{request.knowledge_base_id}].')

    # 修改文档库中当前知识库id下的文档状态为SUCCESS
    try:
        await modify_document_status_by_knowledge_base_id(
            db,
            request.knowledge_base_id,
            DocumentStatus.SUCCESS.value,
        )
    except Exception as e:
        logger.error(f'{task_id} Fail to modify document status to {DocumentStatus.SUCCESS.value} for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to modify document status for:{type(e).__name__}: {str(e)}.'
        )

    logger.debug(f'{task_id} Done.')

    # 返回响应
    return ApiRagResponseModel(
        code=1000,
        msg=f'{task_id} Success.',
        data=None,
        usage=usage.to_dict() if usage else None,
    )
