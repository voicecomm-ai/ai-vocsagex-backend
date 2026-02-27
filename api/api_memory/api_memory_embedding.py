from typing import List, Dict
import traceback
import json

from pydantic import BaseModel, Field, model_validator
from fastapi import Request, Body

from api.api import app, get_api_client_tag
from api.base_model import GeneratorRequestModel
from api.base_model import ResponseModel as ApiMemoryResponseModel
from logger import get_logger
from core.memory.memory import Memory

logger = get_logger('api')

class ApiMemoryRequestModel(GeneratorRequestModel):
    content: str = Field(..., description="记忆内容")

    @model_validator(mode="after")
    def check_input(self) -> "ApiMemoryRequestModel":
        if not self.content:
            raise ValueError("Empty content.")
        
        return self
    
class ApiMemoryDataModel(BaseModel):
    vector: List[float] = Field(..., description="向量")

@app.post(
    path="/Voicecomm/VoiceSageX/Memory/Embedding",
    response_model=ApiMemoryResponseModel,
    response_model_exclude_none=False,
)
async def handler(conn: Request, body: Dict = Body(...)):
    tag, task_id = get_api_client_tag(conn)
    logger.debug(f'{task_id} {tag}')

    # 格式校验
    try:
        logger.debug(f'{task_id} Request body:\n{json.dumps(body, indent=4, ensure_ascii=False)}')
        request = ApiMemoryRequestModel.model_validate(body)
    except Exception as e:
        logger.error(f"{task_id} Fail to validate pydantic instance for:\n{traceback.format_exc()}")
        return ApiMemoryResponseModel(
            code=2000,
            msg=f'{task_id}: The request body field is incorrect for:{type(e).__name__}: {str(e)}.'
        )
    
    try:
        vector, usage = await Memory.embedding(
            request.content, 
            request.model_instance_provider, 
            request.model_instance_config,
        )
    except Exception as e:
        logger.error(f"{task_id} Fail to embedding for:\n{traceback.format_exc()}")
        return ApiMemoryResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to embedding for:{type(e).__name__}: {str(e)}.'
        )
    
    logger.debug(f"{task_id} Done.")
    
    return ApiMemoryResponseModel(
        code=1000,
        msg=f"{task_id} Success.",
        done=True,
        data=ApiMemoryDataModel(
            vector=vector
        ).model_dump(),
        usage=usage,
    )