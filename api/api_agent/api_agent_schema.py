'''
    智能体 生成智能体重建信息
'''

from typing import Optional, Dict, List, Literal
import traceback
import json

from pydantic import BaseModel, Field, model_validator
from fastapi import Request, Body

from api.api import app, get_api_client_tag
from api.base_model import ModelModel, MemoryInfoModelBaseApp, AgentSchemaModel
from logger import get_logger
from core.rag.retriever.retrieve_processor import RetrieveConfig
from core.rag.utils.rag_utils import escape_text
from core.agent.memory_agent import MemoryAgent
from api.base_model import McpConfigModel
from api.base_model import ResponseModel as ApiAgentResponseModel
from api.base_utils import compress_json, decompress_json

logger = get_logger('api')

class ApiAgentRequestModel(BaseModel):
    agent_name: str = Field(..., description="智能体名称")
    agent_description: str = Field(..., description="智能体描述")
    chat_model_instance_provider: str = Field(..., description='CHAT模型供应商')
    chat_model_instance_config: Dict = Field(..., description='CHAT模型配置信息')
    chat_model_parameters: Optional[Dict] = Field(default_factory=dict, description='CHAT模型调用参数')
    system_prompt: str = Field(..., description='系统提示词')
    chat_history_depth: Optional[int] = Field(None, description='聊天历史最大轮次')
    is_memory: bool = Field(False, description='是否开启长期记忆')
    memory_info: Optional[MemoryInfoModelBaseApp] = Field(None, description='长期记忆基于应用的信息')
    memory_model: Optional[ModelModel] = Field(None, description="记忆的向量模型")
    agent_max_iterations: Optional[int] = Field(30, description='agent迭代最大次数')
    knowledge_base_list: Optional[List[RetrieveConfig]] = Field(default_factory=list, description='知识库检索列表')
    knowledge_recall_config: Optional[Dict] = Field(None, description='知识库多路召回设置')
    mcp_config: Optional[Dict[str,McpConfigModel]] = Field(None, description='MCP配置列表')
    agent_mode: Optional[Literal['function_call', 'react']] = Field('function_call', description='智能体模式：function_call(默认)或react')
    show_thinking: Optional[bool] = Field(False, description='是否显示思考过程')

    @model_validator(mode='after')
    def check_input(self) -> "ApiAgentRequestModel":
        if not self.agent_name:
            raise ValueError("`agent_name` is necessary.")
        if not self.agent_description:
            raise ValueError("`agent_description` is necessary.")

        if self.chat_history_depth is None or self.chat_history_depth < 0:
            self.chat_history_depth = 0

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

class DataModel(BaseModel):
    '''
    ApiAgentResponseModel.data 的具体结构
    '''
    schema_str: str = Field(..., description="智能体重建schema")

@app.post(
    path='/Voicecomm/VoiceSageX/Agent/Schema',
    response_model=ApiAgentResponseModel,
    response_model_exclude_none=False
)
async def handler(conn: Request, body: Dict = Body(...)):
    tag, task_id = get_api_client_tag(conn)
    logger.debug(f'{task_id} {tag}')

    # 格式校验
    try:
        logger.debug(f'{task_id} Request body:\n{json.dumps(body, indent=4, ensure_ascii=False)}')
        request = ApiAgentRequestModel.model_validate(body)
    except Exception as e:
        logger.error(f"{task_id} Fail to validate pydantic instance for:\n{traceback.format_exc()}")
        return ApiAgentResponseModel(
            code=2000,
            msg=f'{task_id}: The request body field is incorrect for:{type(e).__name__}: {str(e)}.'
        )

    # 生成 schema
    try:
        agent_type=MemoryAgent.__name__

        agent_schema = AgentSchemaModel(
            agent_name=request.agent_name, 
            agent_description=request.agent_description, 
            agent_type=agent_type, 
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
                info=request.memory_info if request.is_memory else None, 
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
        )

        logger.debug(f"{task_id} Agent schema:\n{agent_schema.model_dump_json(indent=4)}")

        schema_str = compress_json(agent_schema.model_dump())
    except:
        logger.error(f'{task_id} Fail to generate agent schema for:\n{traceback.format_exc()}')
        return ApiAgentResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to generate agent schema for:{type(e).__name__}: {str(e)}.'
        )

    return ApiAgentResponseModel(
        code=1000, 
        msg=f"{task_id} Success.", 
        data=DataModel(
            schema_str=schema_str, 
        ).model_dump()
    )
