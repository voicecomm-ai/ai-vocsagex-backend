from typing import Dict, Optional, List
import json
from uuid import uuid4

from jinja2 import Template, Undefined

def generate_query_json_str(querys: List[str]) -> str:
    return json.dumps(querys, ensure_ascii=False, indent=4)

def generate_categories_json_str(category_list: List[Dict]) -> str:
    return json.dumps(category_list, ensure_ascii=False, indent=4)

def generate_instruction_json_str(instructions: List[str]) -> str:
    return json.dumps(instructions, ensure_ascii=False, indent=4)

def generate_category_id() -> str:
    return str(uuid4())

def render_categories(category_list: List[str], category_arguments: Optional[Dict]) -> List[str]:
    arguments = category_arguments or {}
    results = []

    for category in category_list:
        s = Template(
            category, undefined=Undefined
        ).render(**arguments)
        results.append(s)
    return results

def render_instruction(instructions: List[str], instruction_arguments: Dict) -> List[str]:
    arguments = instruction_arguments or {}
    results = []

    for instruction in instructions:
        s = Template(
            instruction, undefined=Undefined
        ).render(**arguments)
        results.append(s)
    return results

def generate_category_object(category_list: List[str]) -> List[Dict]:
    results = []
    
    for category in category_list:
        results.append(
            {
                "category_id": generate_category_id(),
                "category_name": category,
            }
        )
    return results