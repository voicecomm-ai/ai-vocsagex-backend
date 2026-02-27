'''
    节点操作：LLM调用
'''

from typing import Dict, List, Optional, Sequence, Literal

from pydantic import Field, BaseModel, model_validator
from jsonschema.validators import validator_for

from core.node.llm_invoker.llm_invoker import allm_invoke
from api.base_model import NodeRequestModel, GeneratorRequestModel
from api.api import create_node_handler
from logger import get_logger

logger = get_logger('api')

class Message(BaseModel):
    type: Literal['system', 'human', 'ai'] = Field(..., description='消息角色')
    content: str = Field(..., description='消息内容')

class ApiNodeRequestModel(NodeRequestModel, GeneratorRequestModel):
    model_parameters: Optional[Dict] = Field(default_factory=dict, description='CHAT模型调用参数')
    stream: bool = Field(False, description='是否流式')

    prompt_messages: Sequence[Message] = Field(..., description='输入消息')
    input_arguments: Optional[Dict] = Field(default_factory=dict, description='输入参数')

    is_history: bool = Field(False, description='是否启用记忆(聊天历史)')
    chat_history: Optional[List] = Field(default_factory=list, description='聊天历史')
    chat_history_depth: Optional[int] = Field(50, ge=0, description='聊天历史最大轮次')

    is_vision: bool = Field(False, description='是否启用视觉')
    vision_images: Optional[List] = Field(default_factory=list, description='视觉文件内容列表')
    vision_resolution: Optional[Literal['low', 'high']] = Field(None, description='视觉分辨率')

    is_structured_output: bool = Field(False, description='是否结构化输出')
    structured_output_schema: Optional[Dict] = Field(default_factory=dict, description='结构化输出的JSON schema')
    
    @model_validator(mode='after')
    def validate_prompt_messages(self) -> 'ApiNodeRequestModel':
        if self.prompt_messages:
            if self.prompt_messages[0].type != 'system':
                raise ValueError("'prompt_messages' does not start with 'system'.") 
        else:
            raise ValueError("Empty 'prompt_messages'.")
        
        # 在系统提示词末尾增加/no_think
        self.prompt_messages[0].content += '\n/no_think'

        return self
    
    @model_validator(mode='after')
    def validate_structured_output(self) -> 'ApiNodeRequestModel':
        if self.is_structured_output:
            try:
                validator_for(
                    self.structured_output_schema
                ).check_schema(
                    self.structured_output_schema
                )
            except Exception as e:
                raise ValueError(f"'structured_output_schema' is invalid for {type(e).__name__}: {str(e)}.")
        return self

    def to_dict(self) -> Dict:
        return self.model_dump()

create_node_handler(
    route='/Voicecomm/VoiceSageX/Node/LLMInvoke',
    request_model=ApiNodeRequestModel,
    func=allm_invoke
)