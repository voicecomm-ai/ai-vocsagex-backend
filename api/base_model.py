from typing import Dict, Optional, Literal, List
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from core.rag.retriever.retrieve_processor import RetrieveConfig, RecallConfig

class ResponseModel(BaseModel):
    code: int = Field(..., description='状态码，成功为1000，失败为2000')
    msg: str = Field(..., description='状态信息')
    done: bool = Field(True, description='是否结束')
    data: Optional[Dict] = Field(None, description='数据')
    usage: Optional[Dict] = Field(None, description='模型tokens用量')

class GeneratorRequestModel(BaseModel):
    model_instance_provider: str = Field(..., description='模型供应商')
    model_instance_config: Dict = Field(..., description='模型配置')

class NodeRequestModel(BaseModel, ABC):
    
    @abstractmethod
    def to_dict(self) -> Dict:
        raise NotImplementedError

class RagRequestModel(BaseModel):
    pass

class ModelModel(BaseModel):
    model_instance_provider: str = Field(..., description='模型加载方式')
    model_instance_config: Dict = Field(..., description='模型配置信息')
    model_parameters: Optional[Dict] = Field(None, description='模型调用参数')

class McpConfigModel(BaseModel):
  transport: Literal['streamable_http', 'stdio', 'sse'] = Field(...,
                                                                description='交互方式')
  url: str = Field(..., description='url')

class MemoryInfoModel(BaseModel):
    application_id: int = Field(..., description="应用id")
    user_id: int = Field(..., description="创建人")
    agent_id: int = Field(..., description="智能体id")
    expired_time: Optional[str] = Field(None, description="长期记忆过期时间")
    data_type: str = Field(..., description="记忆来源类型")

class MemoryInfoModelBaseConv(BaseModel):
    '''
    长期记忆 - 基于会话 的参数
    '''
    user_id: int = Field(..., description="创建人")
    data_type: str = Field(..., description="记忆来源类型")

class MemoryInfoModelBaseApp(BaseModel):
    '''
    长期记忆 - 基于应用 的参数
    '''
    application_id: int = Field(..., description="应用id")
    agent_id: int = Field(..., description="智能体id")
    expired_time: Optional[str] = Field(None, description="长期记忆过期时间")

class AgentSchemaModel(BaseModel):
    class _HistoryModel(BaseModel):
        depth: int = Field(..., description="聊天历史最大轮次")
    
    class _MemoryModel(BaseModel):
        enable: bool = Field(False, description="是否启用长期记忆")
        info: Optional[MemoryInfoModelBaseApp] = Field(None, description="长期记忆使用的信息")
        chat_model: Optional[ModelModel] = Field(None, description="长期记忆使用的文本生成模型")
        embedding_model: Optional[ModelModel] = Field(None, description="长期记忆使用的向量模型")

    class _McpModel(BaseModel):
        enable: bool = Field(False, description="是否启用MCP")
        mcp_dict: Optional[Dict[str, McpConfigModel]] = Field(None, description="MCP配置列表")

    class _KbaseModel(BaseModel):
        enable: bool = Field(False, description="是否启用知识库")
        kb_list: Optional[List[RetrieveConfig]] = Field(None, description="知识库检索设置列表")
        recall: Optional[RecallConfig] = Field(None, description="多路知识库召回设置")

    agent_name: str = Field(..., description="智能体名称")
    agent_description: str = Field(..., description="智能体描述")
    agent_type: str = Field(..., description="智能体类型，即内部智能体类名")
    agent_mode: str = Field(..., description="智能体模式")
    agent_recursion_limit: int = Field(..., description="agent迭代最大次数")
    think: bool = Field(False, description="是否思考")
    system_prompt: str = Field(..., description="系统提示词")
    chat_model: ModelModel = Field(..., description="智能体聊天模型")
    history: _HistoryModel = Field(..., description="聊天历史配置")
    memory: _MemoryModel = Field(..., description="长期记忆配置")
    mcp: _McpModel = Field(..., description="MCP配置")
    kbase: _KbaseModel = Field(..., description="知识库配置")

