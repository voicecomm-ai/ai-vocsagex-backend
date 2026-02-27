from typing import Dict, Any, Optional, Tuple

import httpx

async def aprocess(
    url: str, 
    headers: Dict[str, Any],
    timeout: int, 
    language: str, 
    code: str, 
    **kwargs
) -> Tuple[str, str]:
    payload = {
        "language": language, 
        "code": code, 
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url=url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json().get('data', {})

    return data.get('error', ''), data.get('stdout', '')

def process(
    url: str, 
    headers: Dict[str, Any],
    timeout: int, 
    language: str, 
    code: str, 
    **kwargs
) -> Tuple[str, str]:
    payload = {
        "language": language, 
        "code": code, 
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url=url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json().get('data', {})

    return data.get('error', ''), data.get('stdout', '')