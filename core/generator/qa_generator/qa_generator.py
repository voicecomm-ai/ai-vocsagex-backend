from typing import Dict
import re

from core.generator.qa_generator.prompt_0 import PROMPT_TEMPLATE_GENERATE_QA

_model_parameters = {
    "temperature": 0.01, 
    "num_predict": 2000,
    "seed": 32,
    'reasoning': False, 
}

def _format_split_text(text: str):
    regex = r"Q\d+:\s*(.*?)\s*A\d+:\s*([\s\S]*?)(?=Q\d+:|$)"
    matches = re.findall(regex, text, re.UNICODE)

    return [{"question": q, "answer": re.sub(r"\n\s*", "\n", a.strip())} for q, a in matches if q and a]

def generate_qa(model_instance, language: str, page_content: str, **kwargs) -> Dict:
    output = {
        'done': True,
        'data': {
            'qa': [],
        },
        'usage': {}
    }

    if not page_content or not page_content.strip():
        raise RuntimeError('page_content is empty.')
    if len(language) == 0:
        language = '简体中文'

    result = model_instance.invoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_GENERATE_QA.format_messages(language=language, page_content=page_content),
        model_parameters=_model_parameters,
        stream=False,
    )

    response = result.assistant_message.content

    output['data']['qa'] = _format_split_text(response) if response else []
    output['usage'] = result.usage.to_dict() if result.usage else None

    return output

async def agenerate_qa(model_instance, language: str, page_content: str, **kwargs) -> Dict:
    output = {
        'done': True,
        'data': {
            'qa': [],
        },
        'usage': {}
    }

    if len(page_content) == 0:
        raise RuntimeError('page_content is empty.')
    if len(language) == 0:
        language = '简体中文'

    async for chunk in model_instance.ainvoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_GENERATE_QA.format_messages(language=language, page_content=page_content),
        model_parameters=_model_parameters,
        stream=False,
    ):
        response = chunk.assistant_message.content
        usage = chunk.usage
        break

    qa_list = _format_split_text(response) if response else []

    output['data']['qa'] = qa_list
    output['usage'] = usage.to_dict() if usage else None

    return output