from typing import Dict, List
import traceback
from collections import defaultdict

from core.rag.vectorizer.vectorizer_base import BaseVectorizer
from core.model.model_manager import ModelManager, ModelInstanceType
from logger import get_logger

logger = get_logger('rag')

class NormalVectorizer(BaseVectorizer):
    def __init__(self, **kwargs):
        self.input_args = kwargs
        if 'model_instance_provider' not in self.input_args:
            raise ValueError('NormalVectorizer: missing required parameter: "model_instance_provider".')
        if 'model_instance_config' not in self.input_args:
            raise ValueError('NormalVectorizer: missing required parameter: "model_instance_config".')

        self.model_instance_provider = self.input_args['model_instance_provider']
        self.model_instance_config = self.input_args['model_instance_config']

        if not isinstance(self.model_instance_config, Dict):
            raise ValueError('NormalVectorizer: the required parameter "model_instance_config" type is incorrect.')

        self.model_instance = ModelManager.get_model_instance(
            provider=self.model_instance_provider, 
            model_type=ModelInstanceType.Embedding,
            **(self.model_instance_config),
        )

    def _classify_inputs(self, inputs: List[Dict], key: str='type') -> Dict:
        result = defaultdict(list)
        for idx, item in enumerate(inputs):
            val = item[key]
            result[val].append(idx)
        return result


    async def embedding(self, inputs: List[Dict], **kwargs) -> List[Dict]:
        outputs = []
        for item in inputs:
            item_type = item.get('type')
            item_content = item.get('content')

            if item_type == 'text':
                results = await self.model_instance.ainvoke_text_embedding(
                    texts=[item_content],
                    model_parameters={},
                )
                outputs.append({
                    'type': item_type,
                    'vector': results.embeddings[0],
                    'usage': results.usage.to_dict()
                })
            else:
                outputs.append({})

        # outputs = [{'type': item['type'], 'vector': None, 'usage': None} for item in inputs]
        # inputs_type_map = self._classify_inputs(inputs)

        # if 'text' in inputs_type_map and len(inputs_type_map['text']) > 0:
        #     texts = [inputs[idx]['content'] for idx in inputs_type_map['text']]
        #     results = await self.model_instance.ainvoke_text_embedding(
        #         texts=texts,
        #         model_parameters={'use_mmap': True},
        #     )

        #     for idx, ori_idx in enumerate(inputs_type_map['text']):
        #         outputs[ori_idx]['vector'] = results.embeddings[idx]
        #         outputs[ori_idx]['usage'] = (results.usage // len(texts)).to_dict()
            
        # elif 'image' in inputs_type_map and len(inputs_type_map['image']) > 0:
        #     pass
        # elif 'video' in inputs_type_map and len(inputs_type_map['video']) > 0:
        #     pass

        return outputs