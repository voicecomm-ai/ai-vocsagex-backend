from typing import (Optional, List, Dict)

from langchain_core.messages.base import BaseMessage
from langchain_core.messages import UsageMetadata
from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    ''''
    智能体响应结果结构
    '''
    assistant_response: str
    done: bool = Field(False)
    usage: Optional[UsageMetadata] = Field(None)
    history: List[BaseMessage] = Field(default_factory=list)
    additional_kwargs: Dict = Field(default_factory=dict)
    metadata: Dict = Field(default_factory=dict)

class HistoryDigest(BaseModel):
    '''
    聊天历史摘要结构
    '''
    user_intent: List[str] = Field(..., description="用户意图")
    established_facts: List[str] = Field(..., description="已存在的事实")
    constraints: List[str] = Field(..., description="约束和排除事项")
    tool_calls_summary: List[Dict] = Field(..., description="工具调用摘要")


