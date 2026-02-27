from typing import Dict
from jsonschema import validate
from jsonschema.validators import validator_for

_OUTPUT_FIELD_TYPE_ALLOW = [
    'string', 
    'number', 
    'integer', 
    'object', 
    'boolean', 
]

def check_result(result: Dict, output_schema: Dict):
    # output_schema 校验
    validator_for(output_schema).check_schema(output_schema)

    # out_properties = output_schema.get('properties')

    # # output_schema required 校验
    # defined_fields = set(out_properties.keys())
    # defined_required_fields = set(output_schema.get('required', []))
    # optional_defined_fields = defined_fields - defined_required_fields
    # undefined_required_fields = defined_required_fields - defined_fields
    # if len(undefined_required_fields) > 0:
    #     # 存在未定义的必填字段
    #     raise RuntimeError(f'Undefined required fields exist: {list(undefined_required_fields)}')

    # # output_schema 类型校验
    # for name, details in out_properties.items():
    #     arg_type = details.get('type', '')
    #     if arg_type in _OUTPUT_FIELD_TYPE_ALLOW:
    #         continue
    #     elif arg_type == 'array':
    #         sub_item_type = details.get('items', {}).get('type', '')
    #         if sub_item_type in _OUTPUT_FIELD_TYPE_ALLOW:
    #             continue
    #         else:
    #             raise RuntimeError(f'Unsupported items type in schema: {name} - {sub_item_type}')
    #     else:
    #         raise RuntimeError(f'Unsupported type in schema: {name} - {arg_type}')

    validate(instance=result, schema=output_schema)