from typing import List, Dict
import re

from core.rag.vectorizer.vectorizer_base import BaseVectorizer
from core.rag.vectorizer.vectorizer_normal import NormalVectorizer

_vectorizer_map = {
    'normal': NormalVectorizer
}

class VectorizeProcessor:

    @classmethod
    def get_vectorizer(cls, vectorizer_type: str = 'normal', **kwargs) -> BaseVectorizer:
        if vectorizer_type not in _vectorizer_map.keys():
            raise ValueError(f'Unsupported vertorizer type: {vectorizer_type}')
        return _vectorizer_map[vectorizer_type](**kwargs)

    @classmethod
    async def vectorize(cls, vectorizer: BaseVectorizer, contents: List[str]) -> List[Dict]:
        '''
            生成向量时，勿传入多条，内部已取消了批量生成
        '''
        # 目前暂只支持text
        # inputs = [{'type': 'text', 'content': VectorizeProcessor._remove_image_tag(content)} for content in contents]
        inputs = [{'type': 'text', 'content': content} for content in contents]
        outputs = await vectorizer.embedding(inputs)
        return outputs
    
    @staticmethod
    def _remove_image_tag(content: str) -> str:
        tmp = re.sub(r'!\[.*?\]\([^)]+\)', '', content)
        return tmp if tmp else 'image'