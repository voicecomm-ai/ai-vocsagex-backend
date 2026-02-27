from enum import Enum
from typing import List, Dict, Optional, Literal

from pydantic import BaseModel, Field, model_validator

metadata_types = [
    "string",
    "number",
    "time",
]

string_operators = [
    "=",
    "!=",
    "contains",
    "not contain",
    "starts with",
    "ends with",
    "is null",
    "is not null",
]

number_operators = [
    "=",
    "!=",
    ">",
    "<",
    ">=",
    "<=",
    "is null",
    "is not null",
]

time_operators = [
    "=",
    ">",
    "<",
    "is null",
    "is not null",
]

class MetadataMode(Enum):
    AUTOMATIC = 'AUTOMATIC'
    MANUAL = 'MANUAL'

class ChatModel(BaseModel):
    model_instance_provider: str = Field(..., description='模型供应商')
    model_instance_config: Dict = Field(..., description='模型配置')
    model_parameters: Optional[Dict] = Field(default_factory=dict, description='CHAT模型调用参数')

class MetadataModel(BaseModel):
    metadata_name: str = Field(..., description='元数据名称')
    metadata_type: str = Field(..., description='元数据类型')
    operator_name: Optional[str] = Field(None, description='操作符')
    operator_value: Optional[str] = Field(None, description='操作数')

    @model_validator(mode='after')
    def validate_input(self) -> 'MetadataModel':
        if self.metadata_type not in metadata_types:
            raise ValueError(f"Unsupported metadata type: {self.metadata_type}.")
        return self

class MetadataAutomaticModel(BaseModel):
    chat_model: ChatModel = Field(..., description='CHAT模型信息')
    metadatas: List[MetadataModel] = Field(..., description='元数据信息')
    logical_operator: Literal['AND', 'OR'] = Field('AND', description='条件间的逻辑运算')

class MetadataManualModel(BaseModel):
    metadatas: List[MetadataModel] = Field(..., description='元数据信息')
    logical_operator: Literal['AND', 'OR'] = Field('AND', description='条件间的逻辑运算')

    @model_validator(mode='after')
    def validate_input(self) -> 'MetadataManualModel':
        if not all(metadata.operator_name for metadata in self.metadatas):
            raise ValueError("In manual mode, the operator must be filled in.")
        return self

