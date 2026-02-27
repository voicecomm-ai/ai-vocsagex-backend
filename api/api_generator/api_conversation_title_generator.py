'''
    会话标题生成器
'''

from typing import Dict, Union, AsyncGenerator

from pydantic import Field

from core.generator.conversation_title_generator.conversation_title_generator import agenerate_conversation_title
from api.base_model import GeneratorRequestModel
from api.api import create_generator_handler

class ApiGeneratorRequestModel(GeneratorRequestModel):
    query: str = Field(..., description="用户查询")
    stream: bool = Field(False, description='是否流式')

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "stream": self.stream,
        }

create_generator_handler(
    route='/Voicecomm/VoiceSageX/Generate/ConversationTitle',
    request_model=ApiGeneratorRequestModel,
    func=agenerate_conversation_title
)
