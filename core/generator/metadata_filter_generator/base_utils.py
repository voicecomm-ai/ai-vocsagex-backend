from typing import Dict, List, Any
import json

def generate_metadata_fields_json_str(metadata_fields: List[Dict[str, Any]]) -> str:
    return json.dumps(metadata_fields, ensure_ascii=False, indent=4)

