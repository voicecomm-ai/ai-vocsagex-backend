from typing import Dict, Optional

from core.generator.json_schema_generator.prompt_0 import PROMPT_TEMPLATE_GENERATE_JSON_SCHEMA

async def agenerate_json_schema(
    model_instance,
    model_parameters: Optional[Dict],
    description: str,
    **kwargs
) -> Dict:
    output = {
        'done': True,
        'data': {
            'json_schema': '',
        },
        'usage': {}
    }
    
    parameters = dict(model_parameters or {})
    parameters["reasoning"] = False

    generator = model_instance.ainvoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_GENERATE_JSON_SCHEMA.format_messages(description=description), 
        model_parameters=parameters,
        stream=False,
    )

    async for chunk in generator:
        response = chunk.assistant_message.content
        usage = chunk.usage
        break

    output['data']['json_schema'] = response
    output['usage'] = usage.to_dict() if usage else None

    return output