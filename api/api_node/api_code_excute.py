'''
    节点操作：代码执行
'''

from typing import Dict, Literal

from pydantic import Field, model_validator

from core.node.code_excutor.code_excutor import aexcute_code
from api.base_model import NodeRequestModel
from api.api import create_node_handler
from config.config import get_config
from logger import get_logger

logger = get_logger('api')

class ApiNodeRequestModel(NodeRequestModel):
    language: Literal['python3', 'javascript'] = Field(..., description='代码语言')
    code: str = Field(..., description='执行代码')
    input_variables: Dict = Field(default_factory=dict, description='输入参数')
    output_schema: Dict = Field(..., description='输出参数的JSON schema')

    @model_validator(mode='after')
    def check_language_field(self) -> 'ApiNodeRequestModel':
        if self.language == 'javascript':
            self.language = 'nodejs'
        return self

    def to_dict(self) -> Dict:
        return {
            "language": self.language,
            "code": self.code,
            "input_variables": self.input_variables,
            "output_schema": self.output_schema,
        }

async def handler_func(language: str, code: str, input_variables: Dict, output_schema: Dict) -> Dict:
    try:
        extra_info = get_config().get('component').get('sandbox')
    except:
        import traceback
        logger.error(f'Failed to get dependent info:\n{traceback.format_exc()}')
        raise RuntimeError('Failed to get dependent info.')
    return await aexcute_code(extra_info, language, code, input_variables, output_schema)

create_node_handler(
    route='/Voicecomm/VoiceSageX/Node/CodeExcute',
    request_model=ApiNodeRequestModel,
    func=handler_func,
)