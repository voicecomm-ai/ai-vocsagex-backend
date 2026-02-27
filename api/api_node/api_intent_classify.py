'''
    节点操作：意图分类
'''

from typing import Dict, List, Optional, Literal

from pydantic import Field, model_validator

from api.base_model import NodeRequestModel, GeneratorRequestModel
from api.api import create_node_handler
from core.node.intent_classifier.intent_classifier import aintent_classify


class ApiNodeRequestModel(NodeRequestModel, GeneratorRequestModel):
    model_parameters: Optional[Dict] = Field(default_factory=dict, description='CHAT模型调用参数')

    input_querys: List[str] = Field(..., description='输入查询')

    category_list: List[str] = Field(..., description='类别')
    category_arguments: Optional[Dict] = Field(default_factory=dict, description='类别中使用的参数')

    instruction_list: Optional[List[str]] = Field(None, description='指令')
    instruction_arguments: Optional[Dict] = Field(default_factory=dict, description='指令中使用的参数')

    is_vision: bool = Field(False, description='是否启用视觉')
    vision_images: Optional[List] = Field(default_factory=list, description='视觉文件内容列表')
    vision_resolution: Optional[Literal['low', 'high']] = Field(None, description='视觉分辨率')

    @model_validator(mode='after')
    def validate_input(self) -> 'ApiNodeRequestModel':
        if not self.input_querys:
            raise ValueError("Illegal input query.")
        
        if not self.input_querys[0]:
            raise ValueError("Illegal input query.")

        if not self.category_list:
            raise ValueError("Illegal category list.")

        return self

    def to_dict(self):
        return self.model_dump()

create_node_handler(
    route='/Voicecomm/VoiceSageX/Node/IntentClassify', 
    request_model=ApiNodeRequestModel, 
    func=aintent_classify, 
)