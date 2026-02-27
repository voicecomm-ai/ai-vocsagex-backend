'''
    提示词生成器（已弃用）
'''

from typing import Dict

from pydantic import Field

from core.generator.prompt_generator.prompt_generator import agenerate_prompt
from api.base_model import GeneratorRequestModel
from api.api import create_generator_handler


class ApiGeneratorRequestModel(GeneratorRequestModel):
    instruction: str = Field(..., description='用户指令')
    opening_statement: bool = Field(True, description='是否生成开场白')

    def to_dict(self) -> Dict:
        return {
            "instruction": self.instruction,
            "opening_statement": self.opening_statement,
        }

create_generator_handler(
    route='/Voicecomm/VoiceSageX/Generate/Prompt',
    request_model=ApiGeneratorRequestModel,
    func=agenerate_prompt,
)
