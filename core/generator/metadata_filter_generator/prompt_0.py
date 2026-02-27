'''
    根据查询和当前的元数据字段，生成元数据筛选条件
'''

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate, 
)

'''
    Template variables:
      - query: 输入查询对应的json字符串
      - metadata_fields_json_str: 元数据字段对应的json字符串
'''

_system_message_template = (
    "### Job Description\n"
    "You are a text metadata extract engine that extract text's metadata based on user input and set the metadata value.\n"
    "\n"
    "### Task\n"
    "Your task is to ONLY extract the metadatas that exist in the input text from the provided metadata list.\n"
    'There are three types of metadata: "string", "number", and "time". Different types of metadata support different operators.\n'
    "You need to select the appropriate operator based on the type of metadata.\n"
    'Operators supported by "string" type:\n'
    '   - "=": metadata is the value\n'
    '   - "!=": metadata is not the value\n'
    '   - "contains": metadata contains the value\n'
    '   - "not contain": metadata does not contain the value\n'
    '   - "starts with": metadata starts with the value\n'
    '   - "ends with": metadata ends with the value\n'
    '   - "is null": metadata is null\n'
    '   - "is not null": metadata is not null\n'
    'Operators supported by "number" type:\n'
    '   - "=": metadata is equal to the value\n'
    '   - "!=": metadata is not equal to the value\n'
    '   - ">": metadata is greater than the value\n'
    '   - "<": metadata is less than the value\n'
    '   - ">=": metadata is greater than or equal to the value\n'
    '   - "<=": metadata is less than or equal to the value\n'
    '   - "is null": metadata is null\n'
    '   - "is not null": metadata is not null\n'
    'Operators supported by "time" type:\n'
    '   - "=": metadata is at the moment of the value\n'
    '   - ">": metadata is later than the value\n'
    '   - "<": metadata is earlier than the value\n'
    '   - "is null": metadata is null\n'
    '   - "is not null": metadata is not null\n'
    "\n"
    "### Format\n"
    "The input text is in the variable input_text. Metadata are specified as a list in the variable metadata_fields.\n"
    'Return result in JSON format:\n'
    '   - "metadata_name": name of the metadata\n'
    '   - "metadata_type": type of the metadata\n'
    '   - "operator_value": operand expressed as a string if it exists; otherwise, null.\n'
    '   - "operator_name": operator\n'
    "\n"
    "### Constraint\n"
    'If "metadata_type" is "number" or "time", DO NOT add units to the "operator_value".\n'
    "You can ONLY use the operators I provided ABOVE.\n"
    "You need to use the appropriate operator according to the type.\n"
    "DO NOT include anything other than the JSON array in your response.\n"
    "DO NOT use markdown code blocks (``` or ``` json). Return the raw JSON string directly.\n"
    "\n"
)

_samples_human_message_template_0 = (
    '{{\n'
    '    "input_text": "I want to know which company’s email address test@example.com is?",\n'
    '    "metadata_fields": [\n'
    '        {{\n'
    '            "metadata_name": "filename",\n'
    '            "metadata_type": "string"\n'
    '        }},\n'
    '        {{\n'
    '            "metadata_name": "email",\n'
    '            "metadata_type": "string"\n'
    '        }},\n'
    '        {{\n'
    '            "metadata_name": "phone",\n'
    '            "metadata_type": "string"\n'
    '        }},\n'
    '        {{\n'
    '            "metadata_name": "address",\n'
    '            "metadata_type": "string"\n'
    '        }}\n'
    '    ]\n'
    '}}\n'
)

_samples_ai_message_template_0 = (
    '{{\n'
    '    "metadata_map": [\n'
    '        {{\n'
    '            "metadata_name": "email",\n'
    '            "metadata_type": "string",\n'
    '            "operator_value": "test@example.com",\n'
    '            "operator_name": "="\n'
    '        }}\n'
    '    ]\n'
    '}}\n'
)

_samples_human_message_template_1 = (
'{{\n'
'    "input_text": "What are the movies with a score of more than 9 in 2024?",\n'
'    "metadata_fields": [\n'
'        {{\n'
'            "metadata_name": "name",\n'
'            "metadata_type": "string"\n'
'        }},\n'
'        {{\n'
'            "metadata_name": "year",\n'
'            "metadata_type": "number"\n'
'        }},\n'
'        {{\n'
'            "metadata_name": "rating",\n'
'            "metadata_type": "number"\n'
'        }},\n'
'        {{\n'
'            "metadata_name": "country",\n'
'            "metadata_type": "string"\n'
'        }}\n'
'    ]\n'
'}}\n'
)

_samples_ai_message_template_1 = (
    '{{\n'
    '    "metadata_map": [\n'
    '        {{\n'
    '            "metadata_name": "year",\n'
    '            "metadata_type": "number",\n'
    '            "operator_value": "2024",\n'
    '            "operator_name": "="\n'
    '        }},\n'
    '        {{\n'
    '            "metadata_name": "rating",\n'
    '            "metadata_type": "number",\n'
    '            "operator_value": "9",\n'
    '            "operator_name": ">"\n'
    '        }}\n'
    '    ]\n'
    '}}\n'
)

_human_message_template_0 = (
    '{{\n'
    '    "input_text": "{query}",\n'
    '    "metadata_fields": {metadata_fields_json_str}\n'
    '}}\n'
)

_human_message_template_1 = (
    "{query}\n"
    "/no_think"
)

PROMPT_TEMPLATE_GENERATE_METADATA_FILTER = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessagePromptTemplate.from_template(_system_message_template),
        HumanMessagePromptTemplate.from_template(_samples_human_message_template_0),
        AIMessagePromptTemplate.from_template(_samples_ai_message_template_0),
        HumanMessagePromptTemplate.from_template(_samples_human_message_template_1),
        AIMessagePromptTemplate.from_template(_samples_ai_message_template_1),
        HumanMessagePromptTemplate.from_template(_human_message_template_0),
        HumanMessagePromptTemplate.from_template(_human_message_template_1),
    ]
)