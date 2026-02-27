from typing import Dict, AsyncGenerator

from core.generator.greetings_generator.prompt_0 import PROMPT_TEMPLATE_GENERATE_GREETINGS

_model_parameters = {
    'temperature': 0.01,
    'num_predict': 512,
    'reasoning': False, 
}

def generate_greetings(model_instance, prompt: str, **kwargs) -> Dict:
    output = {
        'done': True,
        'data': {
            'greetings': '',
        },
        'usage': {}
    }

    if len(prompt) == 0:
        raise RuntimeError('Prompt is empty.')
    
    result = model_instance.invoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_GENERATE_GREETINGS.format_messages(prompt=prompt),
        model_parameters=_model_parameters,
        stream=False,
    )

    output['data']['greetings'] = result.assistant_message.content
    output['usage'] = result.usage.to_dict()

    return output

async def agenerate_greetings(model_instance, prompt: str, stream: bool, **kwargs) -> AsyncGenerator:
    output = {
        'done': True,
        'data': {
            'greetings': '',
        },
        'usage': {}
    }

    if len(prompt) == 0:
        raise RuntimeError('Prompt is empty.')
    
    async for chunk in model_instance.ainvoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_GENERATE_GREETINGS.format_messages(prompt=prompt),
        model_parameters=_model_parameters,
        stream=stream,
    ):
        output['data']['greetings'] = chunk.assistant_message.content
        output['usage'] = chunk.usage.to_dict() if chunk.usage else None
        if not stream or chunk.finish_reason:
            output['done'] = True
        else:
            output['done'] = False
        yield output