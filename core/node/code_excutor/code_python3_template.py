from core.node.code_excutor.code_template import CodeTemplate

class Python3CodeTemplate(CodeTemplate):
    _CODE_TEMPLATE = '''
{}

import json
from base64 import b64decode

in_vars_obj = json.loads(b64decode('{}').decode('utf-8'))
output_obj = main(**in_vars_obj)
output_json = json.dumps(output_obj, indent=4)
result = f'{}{{output_json}}{}'
print(result)
'''

    @classmethod
    def get_code_template(cls) -> str:
        return cls._CODE_TEMPLATE