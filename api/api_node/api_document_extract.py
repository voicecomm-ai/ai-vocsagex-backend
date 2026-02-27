'''
    节点操作：文档提取
'''

from typing import List, Union, Dict

from pydantic import Field

from core.node.document_extractor.document_extractor import aextract_document
from api.base_model import NodeRequestModel
from api.api import create_node_handler
from config.config import get_config
from logger import get_logger

logger = get_logger('api')

class ApiNodeRequestModel(NodeRequestModel):
    file: Union[str, List[str]] = Field(..., description='文件')

    def to_dict(self) -> Dict:
        return {
            "file": self.file,
        }

async def handler_func(file: Union[str, List[str]]):
    try:
        path_prefix = get_config().get('dependent_info').get('node').get('document_extract_path_prefix')
    except:
        import traceback
        logger.error(f'Failed to get dependent info:\n{traceback.format_exc()}')
        raise RuntimeError('Failed to get dependent info.')
    return await aextract_document(path_prefix, file)

create_node_handler(
    route='/Voicecomm/VoiceSageX/Node/DocumentExtract',
    request_model=ApiNodeRequestModel,
    func=handler_func,
)