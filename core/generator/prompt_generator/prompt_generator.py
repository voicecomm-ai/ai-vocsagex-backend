from typing import Dict, AsyncGenerator
import ast
import asyncio

from core.generator.prompt_generator.prompt_0 import PROMPT_TEMPLATE_GENERATE_PROMPT
from core.generator.prompt_generator.prompt_1 import PROMPT_TEMPLATE_EXTRACT_VARIABLES
from core.generator.prompt_generator.prompt_2 import PROMPT_TEMPLATE_GENERATE_OPENING_STATEMENT

_model_parameters = {
    'temperature': 0.01,
    'num_predict': 512,
    'reasoning': False, 
}

def generate_prompt(model_instance, instruction: str, opening_statement: bool = True, **kwargs) -> Dict:
    output = {
        'done': True,
        'data': {
            'prompt': '',
            'variables': [],
            'opening_statement': '',
        },
        'usage': {}
    }

    if len(instruction) == 0:
        raise RuntimeError('Instruction is empty.')

    # 1. 生成提示词
    result0 = model_instance.invoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_GENERATE_PROMPT.format_messages(instruction=instruction),
        model_parameters=_model_parameters,
        stream=False,
    )
    prompt = result0.assistant_message.content
    usage = result0.usage

    # 2. 提取变量
    result1 = model_instance.invoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_EXTRACT_VARIABLES.format_messages(prompt=prompt),
        model_parameters=_model_parameters,
        stream=False,
    )
    variables = ast.literal_eval(result1.assistant_message.content)
    usage += result1.usage

    # 3. 生成开场白
    if opening_statement:
        result2 = model_instance.invoke_text_chat(
            prompt_messages=PROMPT_TEMPLATE_GENERATE_OPENING_STATEMENT.format_messages(prompt=prompt, instruction=instruction),
            model_parameters=_model_parameters,
            stream=False,
        )
        opening_statement = result2.assistant_message.content
        usage += result2.usage
        output['data']['opening_statement'] = opening_statement

    output['data']['prompt'] = prompt
    output['data']['variables'] = variables
    output['usage'] = usage.to_dict()

    return output

async def agenerate_prompt(model_instance, instruction: str, opening_statement: bool = True, **kwargs) -> AsyncGenerator:
    output = {
        'done': True,
        'data': {
            'prompt': '',
            'variables': [],
            'opening_statement': '',
        },
        'usage': {}
    }

    if len(instruction) == 0:
        raise RuntimeError('Instruction is empty.')

    # 1. 生成提示词
    async for chunk in model_instance.ainvoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_GENERATE_PROMPT.format_messages(instruction=instruction),
        model_parameters=_model_parameters,
        stream=False,
    ):
        prompt = chunk.assistant_message.content
        usage = chunk.usage

    # 2. 提取变量
    async for chunk in model_instance.ainvoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_EXTRACT_VARIABLES.format_messages(prompt=prompt),
        model_parameters=_model_parameters,
        stream=False,
    ):
        variables = ast.literal_eval(chunk.assistant_message.content)
        usage += chunk.usage

    # 3. 生成开场白
    if opening_statement:
        async for chunk in model_instance.ainvoke_text_chat(
            prompt_messages=PROMPT_TEMPLATE_GENERATE_OPENING_STATEMENT.format_messages(prompt=prompt, instruction=instruction),
            model_parameters=_model_parameters,
            stream=False,
        ):
            output['data']['opening_statement'] = chunk.assistant_message.content
            usage += chunk.usage
    
    output['data']['prompt'] = prompt
    output['data']['variables'] = variables
    output['usage'] = usage.to_dict()

    yield output