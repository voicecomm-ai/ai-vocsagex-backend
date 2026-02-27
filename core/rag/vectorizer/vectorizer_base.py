from abc import ABC, abstractmethod
from typing import Dict, List

class BaseVectorizer(ABC):

    @classmethod
    def merge_embedding(cls, inputs: List[Dict]) -> Dict:
        '''
            多模态embedding合并
        '''
        raise NotImplementedError

    @abstractmethod
    async def embedding(self, inputs: List[Dict], **kwargs) -> List[Dict]:
        '''
            inputs = [
                {
                    "type": "text / image / video",
                    "content": "文本 / 图片的base64 / 视频的base64"
                }
            ]

            将按输入顺序，生成输出，由于图片、视频占用空间过大，不再携带初始content内容
            ouputs = [
                {
                    "type": "text / image / video / unknown",
                    "vector": List[float],
                    "usage": {
                        "prompt_tokens": int,
                        "completion_tokens": int,
                        "total_tokens": int
                    }
                }
            ]
        '''
        raise NotImplementedError