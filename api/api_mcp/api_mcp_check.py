from typing import Dict
import json
import traceback

from fastapi import Request, Body

from api.api import app, get_api_client_tag
from api.base_model import ResponseModel
from core.mcp.mcp_client import MCPClient
from logger import get_logger

logger = get_logger('api')


@app.post(
    path='/Voicecomm/VoiceSageX/Mcp/Check',
    response_model=ResponseModel,
    response_model_exclude_none=False
)
async def handler(conn: Request, body: Dict = Body(...)):
    tag, task_id = get_api_client_tag(conn)
    logger.debug(f'{task_id} {tag}')
    logger.debug(f'{task_id} Request body:\n{json.dumps(body, indent=4, ensure_ascii=False)}')
    try:
        status = await MCPClient.check_mcp(body)
    except Exception as e:
        logger.error(f'{task_id} Fail to connect to mcp server for:\n{traceback.format_exc()}')
        status = False

    logger.debug(f'{task_id} available: {status}')

    return ResponseModel(
        code=1000,
        msg=f'{task_id} Success.',
        done=True,
        data={
            "available": status
        },
        usage=None
    )
