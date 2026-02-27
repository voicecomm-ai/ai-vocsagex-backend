from typing import Dict, AsyncGenerator

from core.generator.code_generator.prompt_0 import PROMPT_TEMPLATE_GENERATE_PYTHON3_CODE
from core.generator.code_generator.prompt_1 import PROMPT_TEMPLATE_GENERATE_JAVASCRIPT_CODE
from logger import get_logger

logger = get_logger('generator')

# _model_parameters = {
#     'temperature': 0.01,
#     'num_predict': 1024,
# }

_lang_prompt_map = {
    'python3': PROMPT_TEMPLATE_GENERATE_PYTHON3_CODE,
    'javascript': PROMPT_TEMPLATE_GENERATE_JAVASCRIPT_CODE,
}

def generate_code(model_instance, model_parameters: Dict, language: str, instruction: str, **kwargs) -> Dict:
    output = {
        'done': True,
        'data': {
            'code': '',
        },
        'usage': {},
    }

    if language not in _lang_prompt_map.keys():
        raise RuntimeError(f'Language "{language}" is not supported.')

    if len(instruction) == 0:
        raise RuntimeError('Instruction is empty.')
    
    result = model_instance.invoke_text_chat(
        prompt_messages=_lang_prompt_map[language].format_messages(instruction=instruction),
        model_parameters=model_parameters,
        stream=False,
    )

    output['data']['code'] = result.assistant_message.content
    output['usage'] = result.usage.to_dict()

    return output

async def agenerate_code(model_instance, model_parameters: Dict, language: str, instruction: str, stream: bool, **kwargs) -> AsyncGenerator:
    output = {
        'done': True,
        'data': {
            'code': '',
        },
        'usage': {},
    }

    if language not in _lang_prompt_map.keys():
        raise RuntimeError(f'Language "{language}" is not supported.')

    if len(instruction) == 0:
        raise RuntimeError('Instruction is empty.')
    
    sum_code = ''

    parameters = dict(model_parameters)
    parameters["reasoning"] = False

    async for chunk in model_instance.ainvoke_text_chat(
        prompt_messages=_lang_prompt_map[language].format_messages(instruction=instruction),
        model_parameters=parameters,
        stream=stream,
    ):
        output['data']['code'] = chunk.assistant_message.content
        output['usage'] = chunk.usage.to_dict() if chunk.usage else None
        sum_code += chunk.assistant_message.content
        if not stream or chunk.finish_reason:
            output['done'] = True
        else:
            output['done'] = False
        yield output

    logger.debug(f'code:\n{sum_code}')

    # result = await model_instance.ainvoke_text_chat(
    #     prompt_messages=_lang_prompt_map[language].format_messages(instruction=instruction),
    #     model_parameters=_model_parameters,
    #     stream=False,
    # )

    # output['data']['code'] = result.assistant_message.content
    # output['usage'] = result.usage.to_dict()

        