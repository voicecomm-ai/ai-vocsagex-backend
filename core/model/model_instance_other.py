from typing import Dict, Generator, Optional, Union, Sequence, Literal, Any, AsyncGenerator
import traceback

from pydantic import BaseModel, Field, field_validator
import httpx
import requests

from core.model.model_entities import (
    RerankResult,
    TokenUsage,
)
from core.model.model_instance import (
    ModelInstance, 
    ModelInstanceType, 
    ModelInstanceRequestTimeout,
)
from core.model.model_manager import register_model_provider

class ModelConfigOtherRerank(BaseModel):
    model_name: str = Field(..., description='模型名称')
    base_url: str = Field(..., description='基础url')
    context_length: Optional[int] = Field(None, description='模型上下文长度')

    @field_validator("base_url", mode='before')
    def remove_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    class Config:
        extra = 'allow'

@register_model_provider('other')
class ModelInstanceOther(ModelInstance):
    def __init__(self, model_type: ModelInstanceType, **kwargs):
        self.model_type = model_type
        self.model_config_raw = kwargs
        self.model_config = None

        if model_type == ModelInstanceType.Rerank:
            self.model_config = ModelConfigOtherRerank.model_validate(kwargs)
        else:
            raise NotImplementedError(f'{self.__class__.__name__}: {model_type.value} hasn\'t implemented.')

    def invoke_rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        return_documents: bool = True,
        model_parameters: Optional[Dict] = None,
        timeout: Optional[ModelInstanceRequestTimeout] = None,
    ) -> RerankResult:
        if not isinstance(self.model_config, ModelConfigOtherRerank):
            raise RuntimeError(f"{self.__class__.__name__}: current model is not Rerank type")

        # other加载方式的rerank模型，不再追加route
        url = f'{self.model_config.base_url}'

        headers = {
            'Content-Type': 'application/json'
        }

        payload = {
            'model': self.model_config.model_name,
            'query': query,
            'top_n': top_k,
            'documents': documents,
            'return_documents': return_documents,
        }

        timeout_tuple = timeout.to_tuple() if timeout else ModelInstanceRequestTimeout().to_tuple()

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout_tuple,)
        except Exception as e:
            raise RuntimeError(f'Request to {url} failed : {type(e).__name__} - {str(e)}')
        
        if not response.ok:
            raise RuntimeError(f'Request failed: {response.status_code} - {response.text[:100]}')

        response_json = response.json()
        
        return RerankResult(
            model=response_json['model'],
            results=[
                RerankResult.ResultItem(
                    index=item['index'],
                    score=item['relevance_score'],
                    document=item['document'] if return_documents else None
                )
                for item in response_json['results']
            ],
            usage=TokenUsage(
                prompt_tokens=response_json['usage']['total_tokens']
            )
        )
    
    async def ainvoke_rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        return_documents: bool = True,
        model_parameters: Optional[Dict] = None,
        timeout: Optional[ModelInstanceRequestTimeout] = None,
    ) -> RerankResult:
        if not isinstance(self.model_config, ModelConfigOtherRerank):
            raise RuntimeError(f"{self.__class__.__name__}: current model is not Rerank type")

        # other加载方式的rerank模型，不再追加route
        url = f'{self.model_config.base_url}'

        headers = {
            'Content-Type': 'application/json'
        }

        payload = {
            'model': self.model_config.model_name,
            'query': query,
            'top_n': top_k,
            'documents': documents,
            'return_documents': return_documents,
        }

        timeout_tuple = timeout.to_tuple() if timeout else ModelInstanceRequestTimeout().to_tuple()
        # timeout_tuple 是 (connect, read)，httpx 要转换成 seconds 或 httpx.Timeout 实例
        timeout_seconds = max(timeout_tuple)  # 简化起见取最大值作为总 timeout

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout_seconds,
                )
        except Exception as e:
            raise RuntimeError(f'Request to {url} failed : {type(e).__name__} - {str(e)}')

        if not response.status_code == 200:
            raise RuntimeError(f'Request failed: {response.status_code} - {response.text[:100]}')

        response_json = response.json()

        return RerankResult(
            model=response_json['model'],
            results=[
                RerankResult.ResultItem(
                    index=item['index'],
                    score=item['relevance_score'],
                    document=item['document'] if return_documents else None
                )
                for item in response_json['results']
            ],
            usage=TokenUsage(
                prompt_tokens=response_json['usage']['total_tokens']
            )
        )