'''
    检索，供召回测试使用
'''

from typing import Dict, Optional
import json
import traceback
import copy

from pydantic import BaseModel, Field, model_validator
from fastapi import Request, Body

from core.rag.retriever.retrieve_processor import (
    RetrieveConfig, 
    RetrieveProcessor, 
)
from api.base_model import ResponseModel as ApiRagResponseModel
from api.api import app, get_api_client_tag
from logger import get_logger

logger = get_logger('api')

class ApiRagRequestModel(BaseModel):
    query: str = Field(..., description='用户查询')
    knowledge_base_config: RetrieveConfig = Field(..., description='知识库检索信息')

    @model_validator(mode='after')
    def validate_input(self) -> 'ApiRagRequestModel':
        if not self.query:
            raise ValueError("Illegal query.")
        
        return self

@app.post(
    path='/Voicecomm/VoiceSageX/Rag/Retrieve',
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
    
    try:
        logger.debug(f'{task_id} retrieving...')
        documents, usage = await RetrieveProcessor.retrieve(
            query=request.query, 
            retrieve_config=request.knowledge_base_config, 
            is_metadata_filter=False,
            metadata_mode=None,
            metadata_info=None, 
        )
        logger.debug(f'{task_id} retrieved.')
    except Exception as e:
        logger.error(f'{task_id} Fail to retrieve for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to get required instance for:{type(e).__name__}: {str(e)}.'
        )

    results = []
    for document in documents:
        metadata = copy.deepcopy(document.metadata)
        metadata.pop('context_content')

        # 当是父子分段时，修改子段metadata的idx为f_idx
        if 'f_idx' in metadata:
            metadata['s_idx'] = metadata['idx']
            metadata['idx'] = metadata['f_idx']

        item = {
            "content": document.metadata.get('context_content', ''),
            "title": document.metadata.get('title', ''),
            "url": metadata.get('source', ''),
            "icon": "",
            "metadata": metadata,
        }

        results.append(item)

    response = ApiRagResponseModel(
        code=1000, 
        msg=f'{task_id} Success.',
        done=True,
        data={
            "result": results,
        },
        usage=usage,
    )

    logger.debug(f'{task_id} Response:\n{response.model_dump_json(indent=4)}')
    return response