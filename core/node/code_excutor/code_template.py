from abc import ABC, abstractmethod
from typing import Dict, Any
import json
from base64 import b64encode
import re

class CodeTemplate(ABC):
    _result_tag: str = '<<RESULT_TAG>>'

    @classmethod
    @abstractmethod
    def get_code_template(cls) -> str:
        '''
            获取真实运行的代码
        '''
        pass

    @classmethod
    def serialize_in_vars(cls, in_vars: Dict[str, Any]) -> str:
        in_vars_json_str = json.dumps(in_vars, ensure_ascii=False).encode()
        in_vars_base64_encoded = b64encode(in_vars_json_str).decode("utf-8")
        return in_vars_base64_encoded

    @classmethod
    def get_real_code(cls, code: str, in_vars: Dict[str, Any]) -> str:
        code_template = cls.get_code_template()
        in_vars_str = cls.serialize_in_vars(in_vars)
        real_code = code_template.format(code, in_vars_str, cls._result_tag, cls._result_tag)
        return real_code

    @classmethod
    def extract_real_result(cls, response: str) -> str:
        result = re.search(rf"{cls._result_tag}(.*){cls._result_tag}", response, re.DOTALL)
        if not result:
            raise ValueError("Failed to parse result")
        return result.group(1)
    
    @classmethod
    def get_real_result(cls, response: str) -> Dict:
        try:
            result = json.loads(cls.extract_real_result(response))
        except Exception as e:
            raise RuntimeError('Fail to parse response.')
        
        if not isinstance(result, dict):
            raise RuntimeError('Result must be a dict.')
        
        if not all(isinstance(k, str) for k in result):
            raise ValueError("The keys of result must be strings.")
        
        return result