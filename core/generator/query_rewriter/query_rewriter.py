from typing import Dict, Optional
import json
from logger import get_logger

from core.generator.query_rewriter.prompt_0 import PROMPT_TEMPLATE_REWRITE_QUERY

logger = get_logger('query_rewriter')

_default_model_parameters = {
    "temperature": 0.7,
    'reasoning': False, 
}



async def arewrite_query(
    model_instance, 
    model_parameters: Optional[Dict],
    query: str,
    **kwargs
) -> Dict:
    
    output = {
        "done": True, 
        "data": {
            "query": "",
        }, 
        "usage": {}
    }

    if not model_parameters:
        parameters = _default_model_parameters
    else:
        parameters = dict(model_parameters)
        parameters["reasoning"] = False

    generator = model_instance.ainvoke_text_chat(
        prompt_messages=PROMPT_TEMPLATE_REWRITE_QUERY.format_messages(query=query), 
        model_parameters=parameters,
        stream=False,
    )

    usage = None
    rewritten_query = ""
    
    async for chunk in generator:
        response = chunk.assistant_message.content
        usage = chunk.usage
        
        # 验证响应是否为空
        if not response or not response.strip():
            logger.warning("Model returned empty response for query rewriting")
            break
        
        # 尝试解析 JSON 字符串
        try:
            response_parsed = json.loads(response)
            rewritten_query = response_parsed.get("query", "")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for query rewriting: {e}")
            logger.error(f"Response content (first 1000 chars): {response[:1000]}")
            rewritten_query = response
            logger.warning(f"Using cleaned response as fallback: {rewritten_query[:100]}")
        
        break  # 只处理第一个 chunk

    output["data"]["query"] = rewritten_query
    output["usage"] = usage.to_dict() if usage else None

    return output