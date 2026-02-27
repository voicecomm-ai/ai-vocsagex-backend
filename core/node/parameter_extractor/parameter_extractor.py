from typing import Dict, List, Optional, Literal
import json

from core.model.model_manager import ModelManager, ModelInstanceType
from core.node.parameter_extractor.base_utils import (
    render_str, 
    to_tool,
    extract_args, 
)
from core.node.code_excutor.base_utils import check_result

_default_model_parameters = {
    "temperature": 0.7,
}

async def aextract_parameter(
    model_instance_provider: str,
    model_instance_config: Dict,
    model_parameters: Optional[Dict],
    query: str,
    query_arguments: Optional[Dict],
    args_schema: Dict,
    instruction: str,
    instruction_arguments: Optional[Dict],
    is_vision: bool,
    vision_images: Optional[List],
    vision_resolution: Optional[Literal['low', 'high']],
    reason_mode: Optional[Literal["FunctionCall", "Prompt"]],
    **kwargs
) -> Dict:
    # 获取LLM实例
    model_instance = ModelManager.get_model_instance(
        model_instance_provider,
        ModelInstanceType.LLM,
        **model_instance_config,
    )

    # 参数嵌入
    query = render_str(query, query_arguments)
    instruction = render_str(instruction, instruction_arguments)

    # 推理模式确定
    model_support_func = model_instance_config.get("is_support_function", False)
    if reason_mode == "FunctionCall" and not model_support_func:
        reason_mode = "Prompt"
    elif not reason_mode:
        reason_mode = "FunctionCall" if model_support_func else "Prompt"
 
    if reason_mode == "FunctionCall" and len(query) < 1024:
        # 组装提示词
        from core.node.parameter_extractor.prompt_0 import PROMPT_TEMPLATE_EXTRACT_PARAMETER
        prompt_messages = PROMPT_TEMPLATE_EXTRACT_PARAMETER.format_messages(
            instruction=instruction,
            query=query,
            args_schema_json=json.dumps(args_schema, ensure_ascii=False),
        )

        # 视觉组装
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
            tools=[to_tool(args_schema)],
            stream=False,
        )

        async for chunk in generator:
            parameters = extract_args(chunk.assistant_message)

            check_result(parameters, args_schema)

            output = {
                "done": True,
                "data": {
                    "result": parameters,
                },
                "usage": chunk.usage.to_dict(),
            }
            
            return output
    else:
        # 组装提示词
        from core.node.parameter_extractor.prompt_1 import PROMPT_TEMPLATE_EXTRACT_PARAMETER
        prompt_messages = PROMPT_TEMPLATE_EXTRACT_PARAMETER.format_messages(
            instruction=instruction,
            query=query,
            args_schema_json=json.dumps(args_schema, ensure_ascii=False),
        )

        # 视觉组装
        if is_vision and model_instance_config.get('is_support_vision', False):
            prompt_messages[-1].additional_kwargs['images'] = vision_images

        # 调用
        parameters = dict(model_parameters or {})
        parameters["reasoning"] = False

        generator = model_instance.ainvoke_text_chat(
            prompt_messages=prompt_messages,
            model_parameters=parameters,
            stream=False,
        )

        async for chunk in generator:
            parameters = json.loads(chunk.assistant_message.content)

            check_result(parameters, args_schema)

            output = {
                "done": True,
                "data": {
                    "result": parameters,
                },
                "usage": chunk.usage.to_dict(),
            }
            
            return output


