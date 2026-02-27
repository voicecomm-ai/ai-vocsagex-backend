'''
    预训练模型对外API的内部接口
'''

from typing import Dict, Optional
import traceback
import json

from pydantic import BaseModel, Field
from fastapi import Request, Body
from fastapi.responses import StreamingResponse

from api.api import app, get_api_client_tag
from logger import get_logger
from core.model.model_manager import ModelManager, ModelInstanceType

logger = get_logger('api')

class ApiRequestModel(BaseModel):
    model_instance_type: str = Field(..., description='模型类型')
    model_instance_provider: str = Field(..., description='模型加载方式')
    model_instance_config: Dict = Field(..., description='模型配置信息')
    model_inputs: Dict = Field(..., description='模型输入')
    model_parameters: Optional[Dict] = Field(None, description='模型调用参数')
    stream: bool = Field(False, description='是否流式返回')

class ApiResponseModel(BaseModel):
    code: int = Field(..., description='状态码')
    msg: str = Field(..., description='状态信息')
    done: bool = Field(True, description='是否结束')
    data: Optional[Dict] = Field(None, description='数据')
    usage: Optional[Dict] = Field(None, description='tokens用量')

@app.post(
    path='/Voicecomm/VoiceSageX/Model/PretrainedInvoke',
    response_model=ApiResponseModel,
    response_model_exclude_none=False,
)
async def handler(conn: Request, body: Dict = Body(...)):
    tag, task_id = get_api_client_tag(conn)
    logger.debug(f'{task_id} {tag}')

    # 当为外部部署服务时，下述字段将从info中读取
    if app.state.type == 'outer':
        with open('info.json', 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        body['model_instance_type'] = info['model_instance_type']
        body['model_instance_provider'] = info['model_instance_provider']
        body['model_instance_config'] = info['model_instance_config']

    # 格式校验
    try:
        logger.debug(f'{task_id} Request body:\n{json.dumps(body, indent=4, ensure_ascii=False)}')
        request = ApiRequestModel.model_validate(body)
    except Exception as e:
        logger.error(f'{task_id} Fail to validate pydantic instance for:\n{traceback.format_exc()}')
        return ApiResponseModel(
            code=2000,
            msg=f'{task_id} The request body field is incorrect for:{type(e).__name__}: {str(e)}.'
        )
    
    # 根据类型，构建model instance并调用
    try:
        match request.model_instance_type:
            case ModelInstanceType.LLM.value:
                model_instance = ModelManager.get_model_instance(
                    request.model_instance_provider,
                    ModelInstanceType.LLM,
                    **request.model_instance_config
                )
                
                # 从inputs.messages构建prompt_messages
                from core.model.model_utils import generate_messages_from_dict

                result_generator = model_instance.ainvoke_text_chat(
                    prompt_messages=generate_messages_from_dict(request.model_inputs.get('messages')),
                    model_parameters=request.model_parameters,
                    stream=request.stream,
                )

                if not request.stream:
                    async for i in result_generator:
                        result = i

                    return ApiResponseModel(
                        code=1000,
                        msg=f'{task_id} Success.',
                        data={
                            "assistant_message": result.assistant_message.content,
                        },
                        usage=result.usage.to_dict()
                    )
                else:
                    async def streaming_generator():
                        try:
                            logger.debug(f'{task_id} Stream start.')
                            sum_message = ''
                            async for chunk in result_generator:
                                # logger.debug(f'{task_id} Chunk: {chunk}')
                                sum_message += chunk.assistant_message.content
                                t = ApiResponseModel(
                                    code=1000,
                                    msg=f'{task_id}',
                                    data={
                                        "assistant_message": chunk.assistant_message.content,
                                    },
                                    done=True if chunk.usage else False,
                                    usage=chunk.usage.to_dict() if chunk.usage else None,
                                )
                                yield f"data: {t.model_dump_json()}\n\n"
                            
                            logger.debug(f'{task_id} message:\n{sum_message}')
                        except Exception as e:
                            logger.error(f'{task_id} Streaming error:\n{traceback.format_exc()}')
                            t = ApiResponseModel(code=2000, msg=f'{task_id} {str(e)}', done=True)
                            yield f"event: error\ndata: {t.model_dump_json()}\n\n"
                        finally:
                            logger.debug(f'{task_id} Stream done.')

                    return StreamingResponse(streaming_generator(), media_type="text/event-stream")

            case ModelInstanceType.Embedding.value:
                model_instance = ModelManager.get_model_instance(
                    request.model_instance_provider,
                    ModelInstanceType.Embedding,
                    **request.model_instance_config
                )

                usage = None
                embeddings = []
                for item in request.model_inputs.get('embedding_inputs', {}):
                    item_type = item.get('type')
                    item_content = item.get('content')
                    
                    if item_type == 'text':
                        result = await model_instance.ainvoke_text_embedding(
                            texts=[item_content],
                            model_parameters=request.model_parameters, 
                        )
                        embeddings.append({
                            'type': item_type,
                            'vector': result.embeddings[0]
                        })

                        usage = result.usage if usage is None else usage + result.usage
                    else:
                        embeddings.append({'type': item_type, 'vector': []})

                return ApiResponseModel(
                    code=1000,
                    msg=f'{task_id} Success.',
                    data={
                        'embeddings': embeddings,
                    },
                    usage=usage.to_dict()
                )

            case ModelInstanceType.Rerank.value:
                model_instance = ModelManager.get_model_instance(
                    request.model_instance_provider,
                    ModelInstanceType.Rerank,
                    **request.model_instance_config
                )

                inputs = request.model_inputs

                result = await model_instance.ainvoke_rerank(
                    model_parameters=request.model_parameters, 
                    **inputs
                )

                return ApiResponseModel(
                    code=1000,
                    msg=f'{task_id} Success.',
                    data={
                        'results': result.results,
                    },
                    usage=result.usage.to_dict()
                )

            case ModelInstanceType.ASR.value:
                raise ValueError(f'"{request.model_instance_type}" hasn\'t implemented yet.')
            
            case ModelInstanceType.TTS.value:
                raise ValueError(f'"{request.model_instance_type}" hasn\'t implemented yet.')
            
            case ModelInstanceType.ImageGeneration.value:
                raise ValueError(f'"{request.model_instance_type}" hasn\'t implemented yet.')
            
            case ModelInstanceType.VideoGeneration.value:
                raise ValueError(f'"{request.model_instance_type}" hasn\'t implemented yet.')
            
            case ModelInstanceType.Multimodal.value:
                raise ValueError(f'"{request.model_instance_type}" hasn\'t implemented yet.')
            
            case _:
                raise ValueError(f'"{request.model_instance_type}" hasn\'t supported yet.')
            
    except Exception as e:
        logger.error(f'{task_id} Error for:\n{traceback.format_exc()}')
        return ApiResponseModel(
            code=2000,
            msg=f'{task_id} Error for:{type(e).__name__}: {str(e)}.'
        )