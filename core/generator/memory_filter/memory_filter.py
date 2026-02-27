from typing import Dict, Optional, List

_default_model_parameters = {
    "temperature": 0.7,
}

async def afilter_memory(
    model_instance, 
    model_parameters: Optional[Dict],
    query: str, 
    memories: List[Dict], 
    **kwargs
) -> Dict:
    raise NotImplementedError
    
    output = {
        "done": True, 
        "data": {
            "memory": [],
        }, 
        "usage": {}
    }


    