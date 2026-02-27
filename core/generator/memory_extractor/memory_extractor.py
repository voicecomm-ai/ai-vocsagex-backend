from typing import Dict, Optional, List
import json

from core.generator.memory_extractor.prompt_0 import PROMPT_TEMPLATE_EXTRACT_MEMORY

_default_model_parameters = {
    "temperature": 0.3,
    'reasoning': False, 
}

async def aextract_memory(
    model_instance, 
    model_parameters: Optional[Dict],
    query: str,
    answer: str,
    existing_memory_list: List[str],
    **kwargs
) -> Dict:
    output = {
        "done": True, 
        "data": {
            "memory": [],
        }, 
        "usage": {}
    }

    if not model_parameters:
        parameters = _default_model_parameters
    else:
        parameters = dict(model_parameters)
        parameters["reasoning"] = False


    generator = model_instance.ainvoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_EXTRACT_MEMORY.format_messages(
            query=query, answer=answer, 
            existing_memory_list=json.dumps(existing_memory_list, ensure_ascii=False),
        ), 
        model_parameters=parameters,
        stream=False,
    )

    async for chunk in generator:
        response = chunk.assistant_message.content
        response = json.loads(response)
        usage = chunk.usage
        break

    output["data"]["memory"] = response
    output["usage"] = usage.to_dict() if usage else None

    return output