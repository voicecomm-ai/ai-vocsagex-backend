'''
    终止保存并处理
'''

from typing import Dict
import json
import traceback
import asyncio

from pydantic import Field, BaseModel
from fastapi import Request, Body

from core.rag.utils.sql_operation import (
    delete_vector_by_document_id, 
)
from core.database.database_factory import DatabaseFactory
from api.api import app, get_api_client_tag
from logger import get_logger
from config.config import get_config

logger = get_logger('api')

class ApiRagRequestModel(BaseModel):
    key_id: int = Field(..., description='知识库所在表的主键，对应java的文档id')

class ApiRagResponseModel(BaseModel):
    code: int = Field(..., description='状态码，成功为1000，失败为2000')
    msg: str = Field(..., description='状态信息')

@app.post(path='/Voicecomm/VoiceSageX/Rag/ParseFileStop', response_model=ApiRagResponseModel, response_model_exclude_none=False)
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
    
    async with app.state.rag_tasks_mtx:
        item = app.state.rag_tasks_flag.pop(request.key_id, None)
        if item:
            future, stop_event = item
            stop_event.set()
            future.cancel()
            logger.debug(f'{task_id} notify task to cancel.')
            try:
                await future
            except asyncio.CancelledError:
                logger.warning(f'{task_id} Task cancel.')
            except Exception:
                pass
            logger.debug(f'{task_id} del.')

    # 获取数据库实例
    try:
        database_info = get_config()['dependent_info']['database']
        db = DatabaseFactory.get_database(database_info['type'])
    except Exception as e:
        logger.error(f'{task_id} Fail to get database instance for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
        code=2000,
        msg=f'{task_id} Fail to get database instance.',
    )

    # 异步任务：删除document下所有片段
    async def async_task():
        try:
            await delete_vector_by_document_id(db, request.key_id)
            logger.debug(f'{task_id} Delete vectors.')
        except Exception as e:
            logger.error(f'{task_id} Fail to delete vectors for:\n{traceback.format_exc()}')

    asyncio.create_task(async_task())

    logger.debug(f'{task_id} Done.')

    return ApiRagResponseModel(
        code=1000,
        msg=f'{task_id} Success.',
    )