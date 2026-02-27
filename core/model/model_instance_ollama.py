from typing import Dict, Generator, Optional, Union, Sequence, Literal, Any, AsyncGenerator
import json
import re
import traceback

import httpx
import requests
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages.base import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.tools import BaseTool

from core.model.model_entities import (
    TextChatResult,
    TextChatResultChunk,
    TextEmbeddingResult,
    TokenUsage,
)
from core.model.model_instance import (
    ModelInstance, 
    ModelInstanceType, 
    ModelInstanceRequestTimeout,
)
from core.model.model_utils import (
    convert_message_to_str,
    convert_tool_to_dict_ollama,
    convert_chat_to_aimessage_ollama,
    convert_completion_to_aimessage_ollama,
    remove_think_tag,
    filter_valid_args, 
)
from core.model.model_manager import register_model_provider

class ModelConfigOllamaTextEmbedding(BaseModel):
    model_name: str = Field(..., description='模型名称')
    base_url: str = Field(..., description='基础url')
    context_length: int = Field(..., description='模型上下文长度')
    is_support_vision: bool = Field(..., description='是否支持视觉')

    @field_validator("base_url", mode='before')
    def remove_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    class Config:
        extra = 'allow'

class ModelConfigOllamaLLM(BaseModel):
    model_name: str = Field(..., description='模型名称')
    base_url: str = Field(..., description='基础url')
    llm_type: Literal['completion', 'chat'] = Field('chat', description='模型类型')
    context_length: int = Field(..., description='模型上下文长度')
    max_token_length: int = Field(..., description='最大输出token上限')
    is_support_vision: bool = Field(..., description='是否支持视觉')
    is_support_function: bool = Field(..., description='是否支持函数调用')

    class Config:
        extra = 'allow'

@register_model_provider('ollama')
class ModelInstanceOllama(ModelInstance):
    def __init__(self, model_type: ModelInstanceType, **kwargs):
        super().__init__()
        self.model_type = model_type
        self.model_config_raw = kwargs
        self.model_config = None # type: Union[ModelConfigOllamaLLM, ModelConfigOllamaTextEmbedding]

        if model_type == ModelInstanceType.LLM:
            self.model_config = ModelConfigOllamaLLM.model_validate(kwargs)
        elif model_type == ModelInstanceType.Embedding:
            self.model_config = ModelConfigOllamaTextEmbedding.model_validate(kwargs)
        else:
            raise NotImplementedError(f'{self.__class__.__name__}: {model_type.value} hasn\'t implemented.')

    def to_Embeddings(self, **kwargs) -> Embeddings:
        if self.model_type == ModelInstanceType.Embedding:
            temp_kwargs = filter_valid_args(OllamaEmbeddings, kwargs)
            if "model" in temp_kwargs:
                temp_kwargs.pop("model")
            if "base_url" in temp_kwargs:
                temp_kwargs.pop("base_url")

            return OllamaEmbeddings(
                model=self.model_config.model_name, 
                base_url=self.model_config.base_url, 
                **temp_kwargs
            )
        else:
            raise NotImplementedError(f'{self.__class__.__name__}: {self.model_type.value} hasn\'t implemented "to_Embeddings".')

    def to_BaseChatModel(self, **kwargs) -> BaseChatModel:
        if self.model_type == ModelInstanceType.LLM:
            temp_kwargs = dict(kwargs)
            if "num_ctx" not in temp_kwargs:
                temp_kwargs["num_ctx"] = self.model_config.context_length
            if "repeat_last_n" not in temp_kwargs:
                temp_kwargs["repeat_last_n"] = 128
            if "repeat_penalty" not in temp_kwargs:
                temp_kwargs["repeat_penalty"] = 1.1

            temp_kwargs = filter_valid_args(ChatOllama, temp_kwargs)
            if "model" in temp_kwargs:
                temp_kwargs.pop("model")
            if "base_url" in temp_kwargs:
                temp_kwargs.pop("base_url")

            return ChatOllama(
                model=self.model_config.model_name,
                base_url=self.model_config.base_url,
                **temp_kwargs
            )
        else:
            raise NotImplementedError(f'{self.__class__.__name__}: {self.model_type.value} hasn\'t implemented "to_BaseChatModel".')

    def invoke_text_embedding(
        self, 
        texts: Sequence[str], 
        model_parameters: Optional[Dict] = None,
        timeout: Optional[ModelInstanceRequestTimeout] = None,
    ) -> TextEmbeddingResult:
        if not isinstance(self.model_config, ModelConfigOllamaTextEmbedding):
            raise RuntimeError(f"{self.__class__.__name__}: current model is not Text Embedding type")

        # 获取 embeddings
        parameters = model_parameters or {}
        embeddings = self.to_Embeddings(**parameters)

        response = embeddings.embed_documents(
            texts=texts, 
        )

        return TextEmbeddingResult(
            model=self.model_config.model_name,
            embeddings=response,
            usage=TokenUsage(),
        )

    async def ainvoke_text_embedding(
        self, 
        texts: Sequence[str], 
        model_parameters: Optional[Dict] = None,
        timeout: Optional[ModelInstanceRequestTimeout] = None,
    ) -> TextEmbeddingResult:
        if not isinstance(self.model_config, ModelConfigOllamaTextEmbedding):
            raise RuntimeError(f"{self.__class__.__name__}: current model is not Text Embedding type")

        # 获取 embeddings
        parameters = model_parameters or {}
        embeddings = self.to_Embeddings(**parameters)

        response = await embeddings.aembed_documents(
            texts=texts, 
        )

        return TextEmbeddingResult(
            model=self.model_config.model_name,
            embeddings=response,
            usage=TokenUsage(),
        )

    def invoke_text_chat(
        self, 
        prompt_messages: Sequence[BaseMessage], 
        model_parameters: Optional[Dict] = None, 
        tools: Optional[Sequence[BaseTool]] = None, 
        stop: Optional[Sequence[str]] = None, 
        stream: bool = True,
        **kwargs
    ) -> Union[TextChatResult, Generator]:
        if not isinstance(self.model_config, ModelConfigOllamaLLM):
            raise RuntimeError(f"{self.__class__.__name__}: current model is not LLM type")
        
        # 获取chat_model
        parameters = model_parameters or {}
        chat_model = self.to_BaseChatModel(
            **parameters
        )

        if tools:
            chat_model = chat_model.bind_tools(tools)

        if not stream:
            # 非流
            response = chat_model.invoke(prompt_messages, stop=stop)

            return TextChatResult(
                model=self.model_config.model_name, 
                prompt_messages=prompt_messages, 
                assistant_message=response, 
                usage=TokenUsage.transform(response.usage_metadata), 
            )
        else:
            # 流
            def stream_generator() -> Generator[TextChatResultChunk, None, None]:
                generator = chat_model.stream(prompt_messages, stop=stop)

                index = 0
                for chunk in generator:
                    t_chunk = TextChatResultChunk(
                        model=self.model_config.model_name, 
                        prompt_messages=prompt_messages,
                        index=index, 
                        assistant_message=chunk, 
                        finish_reason=chunk.response_metadata.get("done_reason"), 
                        usage=TokenUsage.transform(chunk.usage_metadata) if chunk.usage_metadata else None, 
                    )

                    index += 1
                    yield t_chunk
            return stream_generator()

    async def ainvoke_text_chat(
        self, 
        prompt_messages: Sequence[BaseMessage], 
        model_parameters: Optional[Dict] = None, 
        tools: Optional[Sequence[BaseTool]] = None, 
        stop: Optional[Sequence[str]] = None, 
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator:
        if not isinstance(self.model_config, ModelConfigOllamaLLM):
            raise RuntimeError(f"{self.__class__.__name__}: current model is not LLM type")

        # 获取chat_model
        parameters = model_parameters or {}
        chat_model = self.to_BaseChatModel(
            **parameters
        )

        if tools:
            chat_model = chat_model.bind_tools(tools)

        if not stream:
            # 非流
            response = await chat_model.ainvoke(prompt_messages, stop=stop)

            yield TextChatResult(
                model=self.model_config.model_name, 
                prompt_messages=prompt_messages, 
                assistant_message=response, 
                usage=TokenUsage.transform(response.usage_metadata), 
            )
        else:
            generator = chat_model.astream(prompt_messages, stop=stop)
            index = 0
            async for chunk in generator:
                t_chunk = TextChatResultChunk(
                    model=self.model_config.model_name, 
                    prompt_messages=prompt_messages,
                    index=index, 
                    assistant_message=chunk, 
                    finish_reason=chunk.response_metadata.get("done_reason"), 
                    usage=TokenUsage.transform(chunk.usage_metadata) if chunk.usage_metadata else None, 
                )
                index += 1
                yield t_chunk
