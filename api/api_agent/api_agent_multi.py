'''
    智能体 多智能体 对话
'''

from typing import Optional, Dict, List, Literal
import traceback
import json
import asyncio

from pydantic import BaseModel, Field, model_validator
from fastapi import Request, Body
from fastapi.responses import StreamingResponse

from api.api import app, get_api_client_tag
from logger import get_logger
from api.base_model import ResponseModel as ApiAgentResponseModel
from api.base_model import ModelModel, MemoryInfoModel
from api.base_utils import build_agent
from core.agent_multi.manager_multi_agent import ManagerMultiAgent
from core.model.model_manager import ModelManager, ModelInstanceType
from core.memory.memory import Memory
from core.agent.base_utils import deserialize_history, serialize_history, usage_to_output
from core.rag.utils.rag_utils import escape_text

logger = get_logger('api')

MULTI_AGENT_CLASS = {
    "Manager": ManagerMultiAgent, 
}

class ApiAgentRequestModel(BaseModel):
    chat_model: ModelModel = Field(..., description="基础聊天模型")
    execute_mode: Literal["Manager", "Collaboration"] = Field(..., description="执行模式")
    chat_history_depth: Optional[int] = Field(None, description="聊天历史最大轮次")
    is_memory: bool = Field(False, description='是否开启长期记忆')
    memory_info: Optional[MemoryInfoModel] = Field(None, description='长期记忆的信息')
    memory_model: Optional[ModelModel] = Field(None, description="记忆的向量模型")
    sub_agent_schemas: List[str] = Field(..., description="子智能体的schema")
    max_retry_times: Optional[int] = Field(None, description="最大重试次数，用于失败重试")

    user_query: str = Field(..., description='用户查询')
    chat_history: Optional[List] = Field(default_factory=list, description='聊天历史')
    sub_agent_args: List[Dict] = Field(..., description="子智能体的系统提示词内的变量，需与子智能体一一对应")

    @model_validator(mode='after')
    def check_input(self) -> "ApiAgentRequestModel":
        if not self.user_query:
            raise ValueError("Empty user query.")
        
        if not self.sub_agent_schemas:
            raise ValueError("No sub agents are available.")

        if len(self.sub_agent_schemas) != len(self.sub_agent_args):
            raise ValueError(f"The quantities of schema[{len(self.sub_agent_schemas)}] and args[{len(self.sub_agent_args)}] of sub_agent are not equal.")
        
        if self.chat_history_depth is None or self.chat_history_depth < 0:
            self.chat_history_depth = 0
        if self.chat_history_depth == 0:
            self.chat_history = []

        if self.is_memory:
            if not self.memory_info:
                raise ValueError("`memory_info` is necessary, when `is_memory` is enabled.")
            if not self.memory_model:
                raise ValueError("`memory_model` is necessary, when `is_memory` is enabled.")
        
        if self.max_retry_times is None:
            self.max_retry_times = 1
    
        return self

# 这个接口的usage存在data中
class DataModel(BaseModel):
    '''
    ApiAgentResponseModel.data 的具体结构
    '''
    assistant_message: str = Field(..., description="响应消息")
    chat_history: List[Dict] = Field(default_factory=list, description="聊天历史")
    additional_kwargs: Dict = Field(default_factory=dict, description="额外附加信息")
    usage: Optional[Dict] = Field(None, description="tokens用量")

@app.post(
    path='/Voicecomm/VoiceSageX/Agent/MultiChat',
)
async def handler(conn: Request, body: Dict = Body(...)):
    tag, task_id = get_api_client_tag(conn)
    logger.debug(f'{task_id} {tag}')

    async def inner_streaming_generator(queue: asyncio.Queue):
        while True:
            # 请求体校验
            try:
                logger.debug(f'{task_id} Request body:\n{json.dumps(body, indent=4, ensure_ascii=False)}')
                request = ApiAgentRequestModel.model_validate(body)
            except Exception as e:
                logger.error(f'{task_id} Fail to validate pydantic instance for:\n{traceback.format_exc()}')
                t = ApiAgentResponseModel(
                    code=2000,
                    msg=f'{task_id}: The request body field is incorrect for:{type(e).__name__}: {str(e)}.'
                )

                await queue.put(f"event: error\ndata: {t.model_dump_json()}\n\n")
                break

            # 重建子智能体
            try:
                sub_agents = []
                for idx, sub_schema in enumerate(request.sub_agent_schemas):
                    sub_agent = await build_agent(
                        task_id=task_id, 
                        schema=sub_schema, 
                        agent_args=request.sub_agent_args[idx], 
                        sub=True, 
                    )

                    if not sub_agent.description.strip():
                        # 跳过无效的子智能体
                        logger.warning(f"{task_id} Ignore agent with invalid description: [{sub_agent.name}]")
                        continue

                    sub_agents.append(sub_agent)
                    logger.debug(f"{task_id} Rebuild sub agent: [{sub_agent.name}]\n{sub_agent.description}")
            except Exception as e:
                logger.error(f'{task_id} Fail to rebuild sub agents for:\n{traceback.format_exc()}')
                t = ApiAgentResponseModel(
                    code=2000,
                    msg=f'{task_id}: Fail to rebuild sub agents for:{type(e).__name__}: {str(e)}.'
                )
                await queue.put(f"event: error\ndata: {t.model_dump_json()}\n\n")
                break

            try:
                # 创建多智能体对象
                multi_agent_class = MULTI_AGENT_CLASS.get(request.execute_mode)
                if multi_agent_class is None:
                    raise ValueError(f"Unsupported execute mode: {request.execute_mode}")

                multi_agent = multi_agent_class(
                    name=task_id, 
                    description="A multi agent for temporarily handling some tasks.", 
                    depth=request.chat_history_depth, 
                    retry_times=request.max_retry_times, 
                    reflect_manager=False, 
                    reflect_sub_agent=False, 
                )

                logger.debug(f"{task_id} Created multi-agent.")
            except Exception as e:
                logger.error(f'{task_id} Fail to create multi-agent for:\n{traceback.format_exc()}')
                t = ApiAgentResponseModel(
                    code=2000,
                    msg=f'{task_id}: Fail to create multi-agent for:{type(e).__name__}: {str(e)}.'
                )
                await queue.put(f"event: error\ndata: {t.model_dump_json()}\n\n")
                break

            # 创建聊天模型
            try:
                chat_model_parameters = request.chat_model.model_parameters or {}
                if "num_ctx" not in chat_model_parameters:
                    chat_model_parameters["num_ctx"] = request.chat_model.model_instance_config.get("context_length", 24576)

                planner_chat_model = ModelManager.get_model_instance(
                    provider=request.chat_model.model_instance_provider,
                    model_type=ModelInstanceType.LLM,  # LLM 模型类型
                    **request.chat_model.model_instance_config
                ).to_BaseChatModel(**chat_model_parameters)

                backup_chat_model = ModelManager.get_model_instance(
                    provider=request.chat_model.model_instance_provider,
                    model_type=ModelInstanceType.LLM,  # LLM 模型类型
                    **request.chat_model.model_instance_config
                ).to_BaseChatModel(reasoning=False, **chat_model_parameters)

                aggregator_chat_model = ModelManager.get_model_instance(
                    provider=request.chat_model.model_instance_provider,
                    model_type=ModelInstanceType.LLM,  # LLM 模型类型
                    **request.chat_model.model_instance_config
                ).to_BaseChatModel(reasoning=False, **chat_model_parameters)

                reflection_chat_model = ModelManager.get_model_instance(
                    provider=request.chat_model.model_instance_provider,
                    model_type=ModelInstanceType.LLM,  # LLM 模型类型
                    **request.chat_model.model_instance_config
                ).to_BaseChatModel(reasoning=False, **chat_model_parameters)

                logger.debug(f"{task_id} Created chat model.")
            except Exception as e:
                logger.error(f'{task_id} Fail to create chat model for:\n{traceback.format_exc()}')
                t = ApiAgentResponseModel(
                    code=2000,
                    msg=f'{task_id}: Fail to create chat model for:{type(e).__name__}: {str(e)}.'
                )
                await queue.put(f"event: error\ndata: {t.model_dump_json()}\n\n")
                break

            # 创建记忆函数
            if request.is_memory:
                try:
                    record_afunc = Memory.create_record_afunc(
                        chat_model_instance_provider=request.chat_model.model_instance_provider, 
                        chat_model_instance_config=request.chat_model.model_instance_config, 
                        chat_model_parameters=request.chat_model.model_parameters, 
                        embedding_model_instance_provider=request.memory_model.model_instance_provider, 
                        embedding_model_instance_config=request.memory_model.model_instance_config, 
                        embedding_model_parameters=request.memory_model.model_parameters, 
                        application_id=request.memory_info.application_id, 
                        user_id=request.memory_info.user_id, 
                        agent_id=request.memory_info.agent_id, 
                        data_type=request.memory_info.data_type, 
                        task_id=task_id, 
                    )
                    retrieve_afunc = Memory.create_retrieve_afunc(
                        chat_model_instance_provider=request.chat_model.model_instance_provider, 
                        chat_model_instance_config=request.chat_model.model_instance_config, 
                        chat_model_parameters=request.chat_model.model_parameters, 
                        embedding_model_instance_provider=request.memory_model.model_instance_provider, 
                        embedding_model_instance_config=request.memory_model.model_instance_config, 
                        embedding_model_parameters=request.memory_model.model_parameters, 
                        application_id=request.memory_info.application_id, 
                        user_id=request.memory_info.user_id, 
                        agent_id=request.memory_info.agent_id, 
                        expired_time=request.memory_info.expired_time,
                        data_type=request.memory_info.data_type, 
                        task_id=task_id, 
                    )

                    logger.debug(f"{task_id} Created memory function.")
                except Exception as e:
                    logger.error(f'{task_id} Fail to create memory function for:\n{traceback.format_exc()}')
                    t = ApiAgentResponseModel(
                        code=2000,
                        msg=f'{task_id}: Fail to create memory function for:{type(e).__name__}: {str(e)}.'
                    )
                    await queue.put(f"event: error\ndata: {t.model_dump_json()}\n\n")
                    break

            # 配置多智能体对象
            try:
                multi_agent.set_chat_model(
                    planner_chat_model=planner_chat_model, 
                    backup_chat_model=backup_chat_model, 
                    aggregator_chat_model=aggregator_chat_model, 
                    reflection_chat_model=reflection_chat_model, 
                )
                multi_agent.set_sub_agents(sub_agents)
                if request.is_memory:
                    multi_agent.set_memory(
                        record_afunc=record_afunc, 
                        retrieve_afunc=retrieve_afunc, 
                    )
                
                multi_agent.compile()

                logger.debug(f"{task_id} Compiled multi-agent.")
            except Exception as e:
                logger.error(f'{task_id} Fail to compile multi-agnet for:\n{traceback.format_exc()}')
                t = ApiAgentResponseModel(
                    code=2000,
                    msg=f'{task_id}: Fail to compile multi-agnetl for:{type(e).__name__}: {str(e)}.'
                )
                await queue.put(f"event: error\ndata: {t.model_dump_json()}\n\n")
                break

            # 处理输入
            try:
                user_query = escape_text(request.user_query)

                if request.chat_history_depth > 0 and request.chat_history is not None:
                    chat_history = deserialize_history(request.chat_history)
                else:
                    chat_history = []

                logger.debug(f'{task_id} Processed input infomation.')
            except Exception as e:
                logger.error(f'{task_id} Fail to process input infomation for:\n{traceback.format_exc()}')
                t = ApiAgentResponseModel(
                    code=2000,
                    msg=f'{task_id}: Fail to process input infomation for:{type(e).__name__}: {str(e)}.'
                )
                await queue.put(f"event: error\ndata: {t.model_dump_json()}\n\n")
                break

            # 调用
            try:
                first_chunk = True
                sum_assistant_response = ""
                gen = multi_agent.ainvoke_stream(
                    query=user_query, 
                    history=chat_history, 
                )

                async for chunk in gen:
                    if first_chunk:
                        logger.debug(f"{task_id} Response first chunk.")
                        first_chunk = False

                    sum_assistant_response += chunk.assistant_response
                    chat_history = serialize_history(chunk.history) if chunk.history else []
                    usage = usage_to_output(chunk.usage) if chunk.usage else None
                    t = ApiAgentResponseModel(
                        code=1000, 
                        msg=f"{task_id} Success.", 
                        done=chunk.done,
                        data={
                            "assistant_message": chunk.assistant_response, 
                            "chat_history": chat_history, 
                            "additional_kwargs": chunk.additional_kwargs, 
                            "usage": usage, 
                        }
                    )

                    await queue.put(f"data: {t.model_dump_json()}\n\n")

                logger.debug(f"{task_id} assistant_response:\n{sum_assistant_response}")
                logger.debug(f"{task_id} history:\n{json.dumps(chat_history, indent=4, ensure_ascii=False)}")
                logger.debug(f"{task_id} usage:\n{json.dumps(usage, indent=4, ensure_ascii=False)}")
            except Exception as e:
                logger.error(f'{task_id} Fail to invoke multi-agent for:\n{traceback.format_exc()}')
                t = ApiAgentResponseModel(
                    code=2000,
                    msg=f'{task_id}: Fail to invoke multi-agent for:{type(e).__name__}: {str(e)}.'
                )
                await queue.put(f"event: error\ndata: {t.model_dump_json()}\n\n")
                break
            except asyncio.CancelledError:
                logger.warning(f"{task_id} Task cancel.")
                break

            break

        await queue.put(None)
        logger.debug(f"{task_id} Generation done.")

    async def streaming_generator():
        queue = asyncio.Queue()

        task = asyncio.create_task(
            inner_streaming_generator(queue)
        )

        try:
            while True:
                if await conn.is_disconnected():
                    task.cancel()
                    break

                item = await queue.get()
                if item is None:
                    break

                yield item

        except asyncio.CancelledError:
            task.cancel()
            raise
        finally:
            logger.debug(f"{task_id} 'streaming_generator' done.")
            task.cancel()

    return StreamingResponse(streaming_generator(), media_type="text/event-stream")
