from typing import Dict, AsyncGenerator, Generator, Union

from core.generator.conversation_title_generator.prompt_0 import PROMPT_TEMPLATE_GENERATE_CONVERSATION_TITLE
from logger import get_logger

logger = get_logger('generator')

_model_parameters = {
    'temperature': 0.01,
    'num_predict': 32,
    'reasoning': False, 
}

def generate_conversation_title(model_instance, query: str, stream: bool, **kwargs) -> Union[Dict, Generator]:
    output = {
        'done': True,
        'data': {
            'conversation_title': '',
        },
        'usage': {}
    }

    if len(query) == 0:
        raise ValueError("Query is empty.")
    
    if stream:
        sum_result = ""
        for chunk in model_instance.invoke_text_chat(
            prompt_messages=PROMPT_TEMPLATE_GENERATE_CONVERSATION_TITLE.format_messages(query=query),
            model_parameters=_model_parameters,
            stream=stream,
        ): 
            output['data']['conversation_title'] = chunk.assistant_message.content
            output['usage'] = chunk.usage.to_dict() if chunk.usage else None
            sum_result += chunk.assistant_message.content
            if chunk.finish_reason:
                output['done'] = True
            else:
                output['done'] = False
            yield output
        logger.debug(f'conversation_title_generator:\n{sum_result}')
    else:
        result = model_instance.invoke_text_chat(
            prompt_messages=PROMPT_TEMPLATE_GENERATE_CONVERSATION_TITLE.format_messages(query=query),
            model_parameters=_model_parameters,
            stream=stream,
        )

        output['data']['conversation_title'] = result.assistant_message.content
        output['usage'] = result.usage.to_dict()

        return output

async def agenerate_conversation_title(model_instance, query: str, stream: bool, **kwargs) -> AsyncGenerator:
    output = {
        'done': True,
        'data': {
            'conversation_title': '',
        },
        'usage': {}
    }

    if len(query) == 0:
        raise ValueError("Query is empty.")
    
    sum_result = ""

    async for chunk in model_instance.ainvoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_GENERATE_CONVERSATION_TITLE.format_messages(query=query),
        model_parameters=_model_parameters,
        stream=stream,
    ):
        output['data']['conversation_title'] = chunk.assistant_message.content
        output['usage'] = chunk.usage.to_dict() if chunk.usage else None
        sum_result += chunk.assistant_message.content
        if not stream or chunk.finish_reason:
            output['done'] = True
        else:
            output['done'] = False
        yield output

    logger.debug(f'conversation_title_generator:\n{sum_result}')