'''
    节点操作：参数提取
'''

from typing import Dict, List, Optional, Literal

from pydantic import Field, model_validator
from jsonschema.validators import validator_for

from core.node.parameter_extractor.parameter_extractor import aextract_parameter
from api.base_model import NodeRequestModel, GeneratorRequestModel
from api.api import create_node_handler
from logger import get_logger

logger = get_logger('api')

class ApiNodeRequestModel(NodeRequestModel, GeneratorRequestModel):
    model_parameters: Optional[Dict] = Field(default_factory=dict, description='CHAT模型调用参数')

    query: str = Field(..., description="用户输入")
    query_arguments: Optional[Dict] = Field(None, description="用户输入中待嵌入的参数")

    args_schema: Dict = Field(..., description="提取参数的JSON Schema")

    instruction: str = Field(..., description="提示词")
    instruction_arguments: Optional[Dict] = Field(None, description="提示词中待嵌入的参数")

    is_vision: bool = Field(False, description="是否启用视觉")
    vision_images: Optional[List] = Field(default_factory=list, description="视觉文件内容列表")
    vision_resolution: Optional[Literal['low', 'high']] = Field(None, description="视觉分辨率")

    reason_mode: Optional[Literal["FunctionCall", "Prompt"]] = Field(None, description="推理模式")

    @model_validator(mode='after')
    def validate_input(self) -> 'ApiNodeRequestModel':
        if not self.args_schema:
            raise ValueError("Illegal args_schema.")

        try:
            validator_for(
                self.args_schema
            ).check_schema(
                self.args_schema
            )
        except Exception as e:
            raise ValueError(f"Invalid args_schema for {type(e).__name__}: {str(e)}.")

        return self

    def to_dict(self):
        return self.model_dump()
    
create_node_handler(
    route="/Voicecomm/VoiceSageX/Node/ParameterExtract",
    request_model=ApiNodeRequestModel,
    func=aextract_parameter,
)