from typing import Dict, Generator, Optional, Union, Sequence, AsyncGenerator

from langchain_core.messages.base import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings.embeddings import Embeddings
from langchain_siliconflow import SiliconFlowEmbeddings
from langchain_siliconflow import ChatSiliconFlow
from langchain_siliconflow.chat_models import DEFAULT_API_BASE
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
    filter_valid_args, 
)
from core.model.model_manager import register_model_provider
from core.model.model_entities import ModelConfigLLM, ModelConfigTextEmbedding

@register_model_provider('siliconflow')
class ModelInstanceSiliconFlow(ModelInstance):
    def __init__(self, model_type: ModelInstanceType, **kwargs):
        super().__init__()
        self.model_type = model_type
        self.model_config_raw = kwargs
        self.model_config = None # type: Union[ModelConfigLLM, ModelConfigTextEmbedding]

        if model_type == ModelInstanceType.LLM:
            self.model_config = ModelConfigLLM.model_validate(kwargs)
        elif model_type == ModelInstanceType.Embedding:
            self.model_config = ModelConfigTextEmbedding.model_validate(kwargs)
        else:
            raise NotImplementedError(f'{self.__class__.__name__}: {model_type.value} hasn\'t implemented.')

    def to_Embeddings(self, **kwargs) -> Embeddings:
        if self.model_type == ModelInstanceType.Embedding:
            f_kwargs = filter_valid_args(SiliconFlowEmbeddings, kwargs)
            if "model" in f_kwargs:
                f_kwargs.pop("model")
            if "siliconflow_api_key" in f_kwargs:
                f_kwargs.pop("siliconflow_api_key")

            return SiliconFlowEmbeddings(
                model=self.model_config.model_name, 
                siliconflow_api_key=self.model_config.apikey or None, 
                **f_kwargs
            )
        else:
            raise NotImplementedError(f'{self.__class__.__name__}: {self.model_type.value} hasn\'t implemented "to_Embeddings".')

    def to_BaseChatModel(self, **kwargs) -> BaseChatModel:
        if self.model_type == ModelInstanceType.LLM:
            temp_kwargs = dict(kwargs)
            if "num_ctx" in temp_kwargs:
                temp_kwargs["max_tokens"] = temp_kwargs.pop("num_ctx")
            if "num_predict" in temp_kwargs:
                temp_kwargs["max_tokens"] = temp_kwargs.pop("num_predict")
            if "max_tokens" not in temp_kwargs:
                temp_kwargs["max_tokens"] = self.model_config.context_length
            if "presence_penalty" not in temp_kwargs:
                temp_kwargs["presence_penalty"] = 0.0

            f_kwargs = filter_valid_args(ChatSiliconFlow, temp_kwargs)
            if "model" in f_kwargs:
                f_kwargs.pop("model")
            if "base_url" in f_kwargs:
                f_kwargs.pop("base_url")
            if "api_key" in f_kwargs:
                f_kwargs.pop("api_key")
            if "reasoning" in f_kwargs:
                f_kwargs.pop("reasoning")

            return ChatSiliconFlow(
                model=self.model_config.model_name,
                base_url=self.model_config.base_url or DEFAULT_API_BASE,
                api_key=self.model_config.apikey or None, 
                **f_kwargs
            )
        else:
            raise NotImplementedError(f'{self.__class__.__name__}: {self.model_type.value} hasn\'t implemented "to_BaseChatModel".')

    def invoke_text_embedding(
        self, 
        texts: Sequence[str], 
        model_parameters: Optional[Dict] = None,
        timeout: Optional[ModelInstanceRequestTimeout] = None,
    ) -> TextEmbeddingResult:
        if not isinstance(self.model_config, ModelConfigTextEmbedding):
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
        if not isinstance(self.model_config, ModelConfigTextEmbedding):
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
        if not isinstance(self.model_config, ModelConfigLLM):
            raise RuntimeError(f"{self.__class__.__name__}: current model is not LLM type")
        
        # 获取chat_model
        parameters = dict(model_parameters) or {}
        parameters["streaming"] = stream
        chat_model: BaseChatModel = self.to_BaseChatModel(
            **parameters
        )

        if tools:
            chat_model = chat_model.bind_tools(tools)

        if not stream:
            # 非流
            response = chat_model.invoke(prompt_messages)

            return TextChatResult(
                model=self.model_config.model_name, 
                prompt_messages=prompt_messages, 
                assistant_message=response, 
                usage=TokenUsage.transform(response.usage_metadata), 
            )
        else:
            # 流
            def stream_generator() -> Generator[TextChatResultChunk, None, None]:
                generator = chat_model.stream(prompt_messages)

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
        if not isinstance(self.model_config, ModelConfigLLM):
            raise RuntimeError(f"{self.__class__.__name__}: current model is not LLM type")

        # 获取chat_model
        parameters = dict(model_parameters) or {}
        parameters["streaming"] = stream

        chat_model: BaseChatModel = self.to_BaseChatModel(
            **parameters
        )

        if tools:
            chat_model = chat_model.bind_tools(tools)

        if not stream:
            # 非流
            response = await chat_model.ainvoke(prompt_messages)

            yield TextChatResult(
                model=self.model_config.model_name, 
                prompt_messages=prompt_messages, 
                assistant_message=response, 
                usage=TokenUsage.transform(response.usage_metadata), 
            )
        else:
            generator = chat_model.astream(prompt_messages)
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
