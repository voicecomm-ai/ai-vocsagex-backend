from typing import Dict, Optional, List, Any
import json

from core.generator.metadata_filter_generator.prompt_0 import PROMPT_TEMPLATE_GENERATE_METADATA_FILTER
from core.generator.metadata_filter_generator.base_utils import generate_metadata_fields_json_str

async def agenerate_metadata_filter(
    model_instance, 
    model_parameters: Optional[Dict], 
    query: str, 
    metadata_fields: List[Dict[str, Any]], 
    **kwargs
) -> Dict:
    '''
    INPUT FORMAT:
        metadata_fields = [
            {
                "metadata_name": "",
                "metadata_type": "",
            },
        ]

    OUTPUT FORMAT:
        output.data.metadata_filter = [
            {
                "metadata_name": "",
                "metadata_type": "",
                "operator_value": "",
                "operator_name": "",
            }
        ]

    ATTENTION:
        Supported metadata type:
            from core.rag.metadata.entities import metadata_types
        Supported operators:
            from core.rag.metadata.entities import (
                string_operators, 
                number_operators, 
                time_operators, 
            )
    '''

    output = {
        'done': True,
        'data': {
            'metadata_filter': [],
        },
        'usage': None
    }

    if not metadata_fields:
        return output

    metadata_fields_json_str = generate_metadata_fields_json_str(metadata_fields)

    parameters = dict(model_parameters or {})
    parameters["reasoning"] = False

    generator = model_instance.ainvoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_GENERATE_METADATA_FILTER.format_messages(
            query=query,
            metadata_fields_json_str=metadata_fields_json_str,
        ), 
        model_parameters=parameters,
        stream=False,
    )

    async for chunk in generator:
        response = chunk.assistant_message.content
        usage = chunk.usage
        break

    json_obj = json.loads(response)
    output['data']['metadata_filter'] = json_obj['metadata_map']
    output['usage'] = usage.to_dict() if usage else None

    return output