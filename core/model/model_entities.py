from typing import Sequence, Optional, List, Dict, Literal

from pydantic import BaseModel, Field
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import AIMessage, UsageMetadata

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }
    
    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens
        )
    
    def __floordiv__(self, other: int) -> "TokenUsage":
        if not isinstance(other, int):
            return NotImplemented
        if other == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return TokenUsage(
            prompt_tokens=self.prompt_tokens // other,
            completion_tokens=self.completion_tokens // other
        )
    
    @classmethod
    def transform(cls, src: UsageMetadata) -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=src["input_tokens"], 
            completion_tokens=src["output_tokens"], 
        )

class ModelConfigTextEmbedding(BaseModel):
    model_name: str = Field(..., description='模型名称')
    base_url: str = Field('', description='基础url')
    apikey: str = Field('', description="外部模型服务的apikey")
    context_length: int = Field(..., description='模型上下文长度')
    is_support_vision: bool = Field(..., description='是否支持视觉')

    class Config:
        extra = 'allow'

class ModelConfigLLM(BaseModel):
    model_name: str = Field(..., description='模型名称')
    base_url: str = Field('', description='基础url')
    apikey: str = Field('', description="外部模型服务的apikey")
    llm_type: Literal['completion', 'chat'] = Field('chat', description='模型类型')
    context_length: int = Field(..., description='模型上下文长度')
    max_token_length: int = Field(..., description='最大输出token上限')
    is_support_vision: bool = Field(..., description='是否支持视觉')
    is_support_function: bool = Field(..., description='是否支持函数调用')

    class Config:
        extra = 'allow'

class TextEmbeddingResult(BaseModel):
    model: str
    embeddings: List[List[float]]
    usage: TokenUsage

class TextChatResultChunk(BaseModel):
    model: str
    prompt_messages: List[BaseMessage] = Field(default_factory=list)

    index: int
    assistant_message: AIMessage
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None

class TextChatResult(BaseModel):
    model: str
    prompt_messages: List[BaseMessage] = Field(default_factory=list)
    assistant_message: AIMessage
    usage: TokenUsage

class RerankResult(BaseModel):
    class ResultItem(BaseModel):
        index: int
        score: float
        document: Optional[Dict]
    
    model: str
    results: List[ResultItem] = Field(default_factory=list)
    usage: TokenUsage