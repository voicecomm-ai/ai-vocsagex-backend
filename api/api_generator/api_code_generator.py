'''
    代码生成器
'''

from typing import Dict, Optional, Literal

from pydantic import Field

from core.generator.code_generator.code_generator import agenerate_code
from api.base_model import GeneratorRequestModel
from api.api import create_generator_handler


class ApiGeneratorRequestModel(GeneratorRequestModel):
    model_parameters: Optional[Dict] = Field(default_factory=dict, description='CHAT模型调用参数')
    language: Literal['python3', 'javascript'] = Field(..., description='代码语言')
    instruction: str = Field(..., description='用户指令')
    stream: bool = Field(False, description='是否流式')

    def to_dict(self) -> Dict:
        return {
            "model_parameters": self.model_parameters,
            "language": self.language,
            "instruction": self.instruction,
            "stream": self.stream,
        }


create_generator_handler(
    route='/Voicecomm/VoiceSageX/Generate/Code',
    request_model=ApiGeneratorRequestModel,
    func=agenerate_code,
)
