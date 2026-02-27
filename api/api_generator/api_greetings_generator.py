'''
    开场白生成器
'''

from typing import Dict

from pydantic import Field

from core.generator.greetings_generator.greetings_generator import agenerate_greetings
from api.base_model import GeneratorRequestModel
from api.api import create_generator_handler

class ApiGeneratorRequestModel(GeneratorRequestModel):
    prompt: str = Field(..., description='提示词')
    stream: bool = Field(False, description='是否流式')

    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "stream": self.stream,
        }

create_generator_handler(
    route='/Voicecomm/VoiceSageX/Generate/Greetings',
    request_model=ApiGeneratorRequestModel,
    func=agenerate_greetings
)