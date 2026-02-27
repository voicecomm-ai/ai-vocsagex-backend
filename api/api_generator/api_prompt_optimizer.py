'''
    提示词优化器
'''

from typing import Dict, Union, AsyncGenerator

from pydantic import Field

from core.generator.prompt_optimizer.prompt_optimizer import aoptimize_prompt
from api.base_model import GeneratorRequestModel
from api.api import create_generator_handler

class ApiGeneratorRequestModel(GeneratorRequestModel):
    prompt: str = Field(..., description='提示词')
    instruction: str = Field('', description='用户指令')
    stream: bool = Field(False, description='是否流式')

    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "instruction": self.instruction,
            "stream": self.stream,
        }

create_generator_handler(
    route='/Voicecomm/VoiceSageX/Generate/OptimizePrompt',
    request_model=ApiGeneratorRequestModel,
    func=aoptimize_prompt
)