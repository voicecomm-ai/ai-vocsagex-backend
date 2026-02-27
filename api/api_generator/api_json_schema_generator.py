'''
    JSON Schema 生成器
'''

from typing import Dict, Optional

from pydantic import Field, model_validator

from core.generator.json_schema_generator.json_schema_generator import agenerate_json_schema
from api.base_model import GeneratorRequestModel
from api.api import create_generator_handler

class ApiGeneratorRequestModel(GeneratorRequestModel):
    model_parameters: Optional[Dict] = Field(default_factory=dict, description='CHAT模型调用参数')

    description: str = Field(..., description='JSON Schema的自然语言描述')

    @model_validator(mode='after')
    def validate_input(self) -> 'ApiGeneratorRequestModel':
        if not self.description:
            raise ValueError('Illegal description.')

        return self

    def to_dict(self) -> Dict:
        return self.model_dump()
    
create_generator_handler(
    route='/Voicecomm/VoiceSageX/Generate/JsonSchema',
    request_model=ApiGeneratorRequestModel,
    func=agenerate_json_schema,
)