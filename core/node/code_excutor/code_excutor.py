from typing import Dict

from core.node.code_excutor.code_template import CodeTemplate
from core.node.code_excutor.code_python3_template import Python3CodeTemplate
from core.node.code_excutor.code_nodejs_template import NodeJsCodeTemplate
from core.node.code_excutor.base_utils import check_result
from core.component.sandbox.processor import process, aprocess

_LANGUAGE_TEMPLATE_MAP = {
    'python3': Python3CodeTemplate,
    'nodejs': NodeJsCodeTemplate,
}

def excute_code(extra_info: Dict, language: str, code: str, input_variables: Dict, output_schema: Dict) -> Dict:
    output = {
        'data': None,
        'usage': None 
    }

    if language not in _LANGUAGE_TEMPLATE_MAP.keys():
        raise RuntimeError(f'Unsupported language: {language}')
    
    TEMPLATE: CodeTemplate = _LANGUAGE_TEMPLATE_MAP.get(language)
    
    real_code = TEMPLATE.get_real_code(code, input_variables)

    error, stdout = process(
        url=extra_info["url"], 
        headers=extra_info.get("headers", {}), 
        timeout=extra_info.get("timeout", 5), 
        language=language, 
        code=real_code, 
    )

    if len(error) > 0:
        raise RuntimeError(f'Code excution failed: {error}')
    
    real_result = TEMPLATE.get_real_result(stdout)

    # check and assign result
    check_result(real_result, output_schema)

    output['data'] = real_result

    return output

async def aexcute_code(extra_info: Dict, language: str, code: str, input_variables: Dict, output_schema: Dict) -> Dict:
    output = {
        'data': None,
        'usage': None 
    }

    if language not in _LANGUAGE_TEMPLATE_MAP.keys():
        raise RuntimeError(f'Unsupported language: {language}')
    
    TEMPLATE: CodeTemplate = _LANGUAGE_TEMPLATE_MAP.get(language)
    
    real_code = TEMPLATE.get_real_code(code, input_variables)

    error, stdout = await aprocess(
        url=extra_info["url"], 
        headers=extra_info.get("headers", {}), 
        timeout=extra_info.get("timeout", 5), 
        language=language, 
        code=real_code, 
    )

    if len(error) > 0:
        raise RuntimeError(f'Code excution failed: {error}')
    
    real_result = TEMPLATE.get_real_result(stdout)

    # check and assign result
    try:
        check_result(real_result, output_schema)
    except Exception as e:
        raise type(e)(f"{str(e)[:500]}") from e

    output['data'] = real_result

    return output