'''
    文档元数据筛选条件 生成器(内部测试用)
'''

from typing import Dict, Optional, List, Literal

from pydantic import Field, model_validator, BaseModel

from core.generator.metadata_filter_generator.metadata_filter_generator import agenerate_metadata_filter
from api.base_model import GeneratorRequestModel
from api.api import create_generator_handler

class MetadataField(BaseModel):
    metadata_field_name: str = Field(..., description='元数据名称')
    metadata_field_id: int = Field(..., description='元数据id')
    metadata_field_type: Literal['string', 'number', 'time'] = Field(..., description='元数据类型')

class ApiGeneratorRequestModel(GeneratorRequestModel):
    model_parameters: Optional[Dict] = Field(default_factory=dict, description='CHAT模型调用参数')

    query: str = Field(..., description='查询')
    metadata_fields: List[MetadataField] = Field(..., description='元数据信息')

    @model_validator(mode='after')
    def validate_input(self) -> 'ApiGeneratorRequestModel':
        if not self.query:
            raise ValueError('Illegal query.')

        return self

    def to_dict(self) -> Dict:
        return self.model_dump()

create_generator_handler(
    route='/Voicecomm/VoiceSageX/Generate/MetadataFilter',
    request_model=ApiGeneratorRequestModel,
    func=agenerate_metadata_filter,
)