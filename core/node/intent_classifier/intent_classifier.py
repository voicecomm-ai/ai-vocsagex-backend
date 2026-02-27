from typing import Dict, List, Optional, Literal
import json

from core.model.model_manager import ModelManager, ModelInstanceType
from core.node.intent_classifier.base_utils import (
    generate_query_json_str, 
    generate_categories_json_str, 
    generate_instruction_json_str, 
    render_categories, 
    render_instruction, 
    generate_category_object, 
)
from core.node.intent_classifier.prompt_0 import PROMPT_TEMPLATE_CLASSIFY_INTENT

async def aintent_classify(
    model_instance_provider: str,
    model_instance_config: Dict,
    model_parameters: Optional[Dict],
    input_querys: List[str],
    category_list: List[str],
    category_arguments: Optional[Dict],
    instruction_list: Optional[List[str]],
    instruction_arguments: Optional[Dict],
    is_vision: bool,
    vision_images: Optional[List],
    vision_resolution: Optional[Literal['low', 'high']],
    **kwargs
) -> Dict:
    # 获取LLM实例
    model_instance = ModelManager.get_model_instance(
        model_instance_provider,
        ModelInstanceType.LLM,
        **model_instance_config,
    )

    # 组装类别中的输入参数并生成对应的object
    category_list = render_categories(category_list, category_arguments)
    category_obj_list = generate_category_object(category_list)

    # 组装指令中的输入参数
    instruction_list = instruction_list or []
    instruction_list = render_instruction(instruction_list, instruction_arguments)

    # 生成对应json
    query_json_str = generate_query_json_str(input_querys)
    categories_json_str = generate_categories_json_str(category_obj_list)
    instruction_json_str = generate_instruction_json_str(instruction_list)

    # 参数嵌入内部提示词
    prompt_messages = PROMPT_TEMPLATE_CLASSIFY_INTENT.format_messages(
        query_json_str=query_json_str,
        categories_json_str=categories_json_str,
        instruction_json_str=instruction_json_str,
        query=''.join(input_querys),
    )

    # 根据视觉组装输入
    if is_vision and model_instance_config.get('is_support_vision', False) and vision_images:
        text = prompt_messages[-1].content
        prompt_messages[-1].content = [
            {
                "type": "text", 
                "text": text,
            }
        ]

        images_content = [
            {
                "type": "image_url", 
                "image_url": image
            }
            for image in vision_images
        ]

        prompt_messages[-1].content.extend(images_content)

    # 调用
    parameters = dict(model_parameters or {})
    parameters["reasoning"] = False

    generator = model_instance.ainvoke_text_chat(
        prompt_messages=prompt_messages,
        model_parameters=parameters,
        stream=False,
    )

    # 返回
    async for chunk in generator:
        category_name = json.loads(chunk.assistant_message.content)['category_name']

        output = {
            'done': True,
            'data': {
                'category': category_name
            },
            'usage': chunk.usage.to_dict()
        }

        return output