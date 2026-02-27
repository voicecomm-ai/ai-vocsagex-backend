'''
    智能体 对话
'''

from typing import Optional, Dict, List, Literal
import traceback
import json
import asyncio

from pydantic import BaseModel, Field, model_validator
from fastapi import Request, Body
from fastapi.responses import StreamingResponse
from jinja2 import Template, Undefined
from langchain.tools import BaseTool

from api.api import app, get_api_client_tag
from api.base_model import ModelModel, MemoryInfoModel, AgentSchemaModel, MemoryInfoModelBaseConv, MemoryInfoModelBaseApp
from logger import get_logger
from core.rag.retriever.retrieve_processor import (
    RetrieveConfig,
)
from core.rag.utils.rag_utils import escape_text
from api.base_model import McpConfigModel
from core.agent.memory_agent import MemoryAgent
from core.agent.base_utils import deserialize_history, serialize_history, usage_to_output
from api.base_utils import build_agent

logger = get_logger('api')

class ApiAgentRequestModel(BaseModel):
    chat_model_instance_provider: str = Field(..., description='CHAT模型供应商')
    chat_model_instance_config: Dict = Field(..., description='CHAT模型配置信息')
    chat_model_parameters: Optional[Dict] = Field(default_factory=dict, description='CHAT模型调用参数')
    system_prompt: str = Field(..., description='系统提示词')
    inputs: Optional[Dict] = Field(default_factory=dict, description='输入变量')
    user_query: str = Field(..., description='用户查询')
    chat_history: Optional[List] = Field(default_factory=list, description='聊天历史')
    chat_history_depth: Optional[int] = Field(None, description='聊天历史最大轮次')
    is_memory: bool = Field(False, description='是否开启长期记忆')
    memory_info: Optional[MemoryInfoModel] = Field(None, description='长期记忆使用的信息')
    memory_model: Optional[ModelModel] = Field(None, description="记忆的向量模型")
    agent_max_iterations: Optional[int] = Field(30, description='agent迭代最大次数')
    knowledge_base_list: Optional[List[RetrieveConfig]] = Field(default_factory=list, description='知识库检索列表')
    knowledge_recall_config: Optional[Dict] = Field(None, description='知识库多路召回设置')
    mcp_config: Optional[Dict[str,McpConfigModel]] = Field(None, description='MCP配置列表')
    agent_mode: Optional[Literal['function_call', 'react']] = Field('function_call', description='智能体模式：function_call(默认)或react')
    show_thinking: Optional[bool] = Field(False, description='是否显示思考过程')

    @model_validator(mode='after')
    def check_input(self) -> "ApiAgentRequestModel":
        if not self.user_query:
            raise ValueError("Empty user query.")
        
        if self.chat_history_depth is None or self.chat_history_depth < 0:
            self.chat_history_depth = 0
        if self.chat_history_depth == 0:
            self.chat_history = []
        
        if self.is_memory:
            if not self.memory_info:
                raise ValueError("`memory_info` is necessary, when `is_memory` is enabled.")
            if not self.memory_model:
                raise ValueError("`memory_model` is necessary, when `is_memory` is enabled.")
        
        if self.agent_max_iterations is None:
            self.agent_max_iterations = 30

        # 不再支持自然语言的ReAct解析
        if self.agent_mode != "function_call":
            self.agent_mode = "function_call"
        
        return self

# 这个接口的usage存在data中
class ApiAgentResponseModel(BaseModel):
    code: int = Field(..., description='状态码，成功为1000，失败为2000')
    msg: str = Field(..., description='状态信息')
    done: bool = Field(True, description='是否结束')
    data: Optional[Dict] = Field(None, description='数据')

@app.post(
    path='/Voicecomm/VoiceSageX/Agent/Chat',
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

            # 构建智能体
            try:
                agent = await build_agent(
                    task_id=task_id, 
                    schema=AgentSchemaModel(
                        agent_name=task_id, 
                        agent_description="An agent for temporarily handling some tasks.", 
                        agent_type=MemoryAgent.__name__, 
                        agent_mode=request.agent_mode, 
                        agent_recursion_limit=request.agent_max_iterations, 
                        think=request.show_thinking, 
                        system_prompt=request.system_prompt, 
                        chat_model=ModelModel(
                            model_instance_provider=request.chat_model_instance_provider, 
                            model_instance_config=request.chat_model_instance_config, 
                            model_parameters=request.chat_model_parameters, 
                        ), 
                        history=AgentSchemaModel._HistoryModel(
                            depth=request.chat_history_depth, 
                        ),
                        memory=AgentSchemaModel._MemoryModel(
                            enable=request.is_memory, 
                            info=MemoryInfoModelBaseApp(
                                application_id=request.memory_info.application_id, 
                                agent_id=request.memory_info.agent_id, 
                                expired_time=request.memory_info.expired_time, 
                            ) if request.is_memory else None, 
                            chat_model=ModelModel(
                                model_instance_provider=request.chat_model_instance_provider, 
                                model_instance_config=request.chat_model_instance_config, 
                                model_parameters=request.chat_model_parameters, 
                                think=False, 
                            ) if request.is_memory else None, 
                            embedding_model=request.memory_model if request.is_memory else None, 
                        ), 
                        mcp=AgentSchemaModel._McpModel(
                            enable=True if request.mcp_config else False, 
                            mcp_dict=request.mcp_config or None, 
                        ), 
                        kbase=AgentSchemaModel._KbaseModel(
                            enable=True if request.knowledge_base_list else False, 
                            kb_list=request.knowledge_base_list or None, 
                            recall=request.knowledge_recall_config or None, 
                        )
                    ), 
                    agent_args=request.inputs, 
                    sub=False, 
                    memory_conv_info=MemoryInfoModelBaseConv(
                        user_id=request.memory_info.user_id, 
                        data_type=request.memory_info.data_type, 
                    ) if request.is_memory else None, 
                )

                logger.debug(f"{task_id} Create agent.")
            except Exception as e:
                logger.error(f'{task_id} Fail to create agent for:\n{traceback.format_exc()}')
                t = ApiAgentResponseModel(
                    code=2000,
                    msg=f'{task_id}: Fail to create agent for:{type(e).__name__}: {str(e)}.'
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

            # 调用智能体
            try:
                first_chunk = True
                sum_assistant_response = ""
                async_gen = agent.ainvoke_stream(query=user_query, history=chat_history)
                
                logger.debug(f"{task_id} Generating...")
                async for chunk in async_gen:
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
                logger.error(f'{task_id} Fail to invoke agent for:\n{traceback.format_exc()}')
                t = ApiAgentResponseModel(
                    code=2000,
                    msg=f'{task_id}: Fail to invoke agent for:{type(e).__name__}: {str(e)}.'
                )
                await queue.put(f"event: error\ndata: {t.model_dump_json()}\n\n")
                break
            except asyncio.CancelledError:
                logger.warning(f"{task_id} Task cancel.")
                break

            break

        await queue.put(None)
        logger.debug(f'{task_id} Generation done.')

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
