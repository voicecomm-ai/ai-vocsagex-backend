from typing import Dict, Generator, Union, AsyncGenerator

from core.generator.prompt_optimizer.prompt_0 import PROMPT_TEMPLATE_OPTIMIZE_PROMPT
from logger import get_logger

logger = get_logger('generator')

_model_parameters = {
    'temperature': 0.01,
    'num_predict': 1024,
    'reasoning': False, 
}

def optimize_prompt(model_instance, prompt: str, instruction: str, **kwargs) -> Union[Dict, Generator]:
    output = {
        'done': True,
        'data': {
            'prompt': '',
        },
        'usage': {}
    }

    if len(prompt) == 0:
        raise RuntimeError('Prompt is empty.')
    
    # if len(instruction) == 0:
    #     instruction = 'Optimize the user-provided prompt structure and improve its content. You need to enhance the task processing steps, provide examples, and specify constraints.'
    instruction = 'Optimize the user-provided prompt structure and improve its content. You need to enhance the task processing steps, provide examples, and specify constraints. ' + instruction

    if kwargs.get('stream', None):
        for chunk in model_instance.invoke_text_chat(
            prompt_messages=PROMPT_TEMPLATE_OPTIMIZE_PROMPT.format_messages(prompt=prompt, instruction=instruction),
            model_parameters=_model_parameters,
            stream=True,
        ):
            output['data']['prompt'] = chunk.assistant_message.content
            output['usage'] = chunk.usage.to_dict() if chunk.usage else None
            if chunk.finish_reason:
                output['done'] = True
            yield output
    else:
        result = model_instance.invoke_text_chat(
            prompt_messages=PROMPT_TEMPLATE_OPTIMIZE_PROMPT.format_messages(prompt=prompt, instruction=instruction),
            model_parameters=_model_parameters,
            stream=False,
        )
        output['data']['prompt'] = result.assistant_message.content
        output['usage'] = result.usage.to_dict()
        return output
    
async def aoptimize_prompt(model_instance, prompt: str, instruction: str, stream: bool, **kwargs) -> AsyncGenerator:
    output = {
        'done': True,
        'data': {
            'prompt': '',
        },
        'usage': {}
    }

    if len(prompt) == 0:
        raise RuntimeError('Prompt is empty.')
    
    # if len(instruction) == 0:
    #     instruction = 'Optimize the user-provided prompt structure and improve its content. You need to enhance the task processing steps, provide examples, and specify constraints.'
    
    instruction = 'Optimize the user-provided prompt structure and improve its content. You need to enhance the task processing steps, provide examples, and specify constraints. ' + instruction

    sum_prompt = ''

    async for chunk in model_instance.ainvoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_OPTIMIZE_PROMPT.format_messages(prompt=prompt, instruction=instruction),
        model_parameters=_model_parameters,
        stream=stream,
    ):
        output['data']['prompt'] = chunk.assistant_message.content
        output['usage'] = chunk.usage.to_dict() if chunk.usage else None
        sum_prompt += chunk.assistant_message.content
        if not stream or chunk.finish_reason:
            output['done'] = True
        else:
            output['done'] = False
        yield output
    
    logger.debug(f'prompt_optimizer:\n{sum_prompt}')