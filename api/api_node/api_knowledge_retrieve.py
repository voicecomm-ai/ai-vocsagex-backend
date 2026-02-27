'''
    节点操作：知识检索
'''

from typing import Dict, List, Optional, Literal, Any

from pydantic import Field, model_validator

from api.base_model import NodeRequestModel
from api.api import create_node_handler
from core.node.knowledge_retriever.knowledge_retriever import aknowledge_retrieve
from core.node.knowledge_retriever.knowledge_retriever import (
    RetrieveConfig,
    RecallConfig,
)
from core.rag.metadata.entities import (
    MetadataAutomaticModel,
    MetadataManualModel,
    MetadataMode,
)

class ApiNodeRequestModel(NodeRequestModel):
    query: str = Field(..., description='')
    knowledge_base_list: List[RetrieveConfig] = Field(..., description='知识库检索列表')
    is_recall: bool = Field(False, description='是否召回')
    knowledge_recall_config: Optional[RecallConfig] = Field(None, description='知识库多路召回设置')
    is_metadata_filter: bool = Field(False, description='是否使用元数据过滤')
    metadata_mode: Optional[Literal['AUTOMATIC', 'MANUAL']] = Field(None, description='元数据过滤模式')
    metadata_info: Optional[Any] = Field(None, description='元数据信息')

    @model_validator(mode='after')
    def validate_input(self) -> 'ApiNodeRequestModel':
        if not self.knowledge_base_list:
            raise ValueError("Illegal knowledge_base_list.")
        if self.is_metadata_filter:
            if not self.metadata_mode:
                raise ValueError("Illegal metadata mode.")
            if not self.metadata_info:
                raise ValueError("Illegal metadata info.")
            if self.metadata_mode == MetadataMode.AUTOMATIC.value:
                self.metadata_info = MetadataAutomaticModel.model_validate(self.metadata_info)
            if self.metadata_mode == MetadataMode.MANUAL.value:
                self.metadata_info = MetadataManualModel.model_validate(self.metadata_info)
        return self
    
    def to_dict(self):
        return {
            "query": self.query,
            "knowledge_base_list": self.knowledge_base_list,
            "is_recall": self.is_recall,
            "knowledge_recall_config": self.knowledge_recall_config,
            "is_metadata_filter": self.is_metadata_filter,
            "metadata_mode": self.metadata_mode,
            "metadata_info": self.metadata_info,
        }

create_node_handler(
    route='/Voicecomm/VoiceSageX/Node/RetrieveKnowledge',
    request_model=ApiNodeRequestModel,
    func=aknowledge_retrieve,
)