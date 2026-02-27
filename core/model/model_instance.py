from typing import Dict, Generator, Optional, Union, Sequence, Tuple, AsyncGenerator
from enum import Enum
from abc import ABC

from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.messages.base import BaseMessage
from langchain.tools import BaseTool

from core.model.model_entities import (
    TextChatResult,
    TextEmbeddingResult,
    RerankResult,
)

class ModelInstanceRequestTimeout(BaseModel):
    connect_timeout: float = 5
    read_timeout: float = 60

    @classmethod
    def from_tuple(cls, timeout_tuple: Tuple[float, float]) -> "ModelInstanceRequestTimeout":
        t1, t2 = timeout_tuple
        return cls(connect_timeout=t1, read_timeout=t2)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.connect_timeout, self.read_timeout)

class ModelInstanceType(Enum):
    LLM = "TextGeneration"
    Embedding = "Embedding"
    Rerank = "Rerank"
    ASR = "ASR"
    TTS = "TTS"
    ImageGeneration = 'ImageGeneration'
    VideoGeneration = 'VideoGeneration'
    Multimodal = 'Multimodal'

class ModelInstance(ABC):

    def to_Embeddings(self, **kwargs) -> Embeddings:
        raise NotImplementedError

    def to_BaseChatModel(self, **kwargs) -> BaseChatModel:
        '''
        将ModelInstance转为BaseChatModel

        Args:
            kwargs: BaseChatModel构造除model和base_url外的额外参数
        '''
        raise NotImplementedError

    def invoke_text_embedding(
        self, 
        texts: Sequence[str], 
        model_parameters: Optional[Dict] = None,
        timeout: Optional[ModelInstanceRequestTimeout] = None,
    ) -> TextEmbeddingResult:
        raise NotImplementedError

    async def ainvoke_text_embedding(
        self, 
        texts: Sequence[str], 
        model_parameters: Optional[Dict] = None,
        timeout: Optional[ModelInstanceRequestTimeout] = None,
    ) -> TextEmbeddingResult:
        raise NotImplementedError

    def invoke_vision_embedding(self):
        raise NotImplementedError

    def invoke_video_embedding(self):
        raise NotImplementedError

    def invoke_text_rerank(self):
        raise NotImplementedError

    def invoke_vision_rerank(self):
        raise NotImplementedError

    def invoke_video_rerank(self):
        raise NotImplementedError

    def invoke_text_chat(
        self, 
        prompt_messages: Sequence[BaseMessage], 
        model_parameters: Optional[Dict] = None, 
        tools: Optional[Sequence[BaseTool]] = None, 
        stop: Optional[Sequence[str]] = None, 
        stream: bool = True,
        **kwargs
    ) -> Union[TextChatResult, Generator]:
        raise NotImplementedError

    async def ainvoke_text_chat(
        self, 
        prompt_messages: Sequence[BaseMessage], 
        model_parameters: Optional[Dict] = None, 
        tools: Optional[Sequence[BaseTool]] = None, 
        stop: Optional[Sequence[str]] = None, 
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator:
        raise NotImplementedError

    def invoke_vision_chat(self):
        raise NotImplementedError

    def invoke_video_chat(self):
        raise NotImplementedError

    def invoke_synth(self):
        raise NotImplementedError

    def invoke_recog(self):
        raise NotImplementedError

    def invoke_rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        return_documents: bool = True,
        model_parameters: Optional[Dict] = None,
        timeout: Optional[ModelInstanceRequestTimeout] = None,
    ) -> RerankResult:
        raise NotImplementedError
    
    async def ainvoke_rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        return_documents: bool = True,
        model_parameters: Optional[Dict] = None,
        timeout: Optional[ModelInstanceRequestTimeout] = None,
    ) -> RerankResult:
        raise NotImplementedError