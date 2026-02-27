from core.node.code_excutor.code_template import CodeTemplate

class NodeJsCodeTemplate(CodeTemplate):
    _CODE_TEMPLATE = '''
{}

var in_vars_obj = JSON.parse(Buffer.from('{}', 'base64').toString('utf-8'))
var args = Object.values(in_vars_obj)
var output_obj = main(...args)
var output_json = JSON.stringify(output_obj)
var result = `{}${{output_json}}{}`
console.log(result)
'''

    @classmethod
    def get_code_template(cls) -> str:
        return cls._CODE_TEMPLATE