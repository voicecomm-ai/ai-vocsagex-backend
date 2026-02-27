'''
    意图分类内部提示词
'''

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate, 
)

'''
    Template variables:
      - query_json_str: 输入查询对应的json字符串
      - categories_json_str: 类别对应的json字符串
      - instruction_json_str: 指令对应的json字符串
'''

_system_message_template = (
    "### Job Description\n"
    "You are a text classification engine that analyzes text data and assigns categories based on user input or automatically determined categories.\n"
    "\n"
    "### Task\n"
    "Your task is to assign one categories ONLY to the input text and only one category may be assigned returned in the output.\n"
    "Additionally, you need to extract the key words from the text that are related to the classification.\n"
    "\n"
    "### Format\n"
    "The input text is in the variable input_text.\n"
    "Categories are specified as a category list with two filed category_id and category_name in the variable categories.\n"
    "Classification instructions may be included to improve the classification accuracy.\n"
    "\n"
    "### Constraint\n"
    "DO NOT include anything other than the JSON array in your response.\n"
    "DO NOT use markdown code blocks (``` or ``` json). Return the raw JSON string directly.\n"
    "/no_think\n"
)


_samples_human_message_template_0 = (
    '{{\n'
    '    "input_text": [\n'
    '        "I recently had a great experience with your company. The service was prompt and the staff was very friendly."\n'
    '    ],\n'
    '    "categories": [\n'
    '        {{\n'
    '            "category_id": "f5660049-284f-41a7-b301-fd24176a711c",\n'
    '            "category_name": "Customer Service"\n'
    '        }},\n'
    '        {{\n'
    '            "category_id": "8d007d06-f2c9-4be5-8ff6-cd4381c13c60",\n'
    '            "category_name": "Satisfaction"\n'
    '        }},\n'
    '        {{\n'
    '            "category_id": "5fbbbb18-9843-466d-9b8e-b9bfbb9482c8",\n'
    '            "category_name": "Sales"\n'
    '        }},\n'
    '        {{\n'
    '            "category_id": "23623c75-7184-4a2e-8226-466c2e4631e4",\n'
    '            "category_name": "Product"\n'
    '        }}\n'
    '    ],\n'
    '    "classification_instructions": [\n'
    '        "classify the text based on the feedback provided by customer"\n'
    '    ]\n'
    '}}\n'
)

_samples_ai_message_template_0 = (
    '{{\n'
    '    "keywords": [\n'
    '        "recently",\n'
    '        "great experience",\n'
    '        "company",\n'
    '        "service",\n'
    '        "prompt",\n'
    '        "staff",\n'
    '        "friendly"\n'
    '    ],\n'
    '    "category_id": "f5660049-284f-41a7-b301-fd24176a711c",\n'
    '    "category_name": "Customer Service"\n'
    '}}\n'
)

_samples_human_message_template_1 = (
    '{{\n'
    '    "input_text": [\n'
    '        "bad service, slow to bring the food"\n'
    '    ],\n'
    '    "categories": [\n'
    '        {{\n'
    '            "category_id": "80fb86a0-4454-4bf5-924c-f253fdd83c02",\n'
    '            "category_name": "Food Quality"\n'
    '        }},\n'
    '        {{\n'
    '            "category_id": "f6ff5bc3-aca0-4e4a-8627-e760d0aca78f",\n'
    '            "category_name": "Experience"\n'
    '        }},\n'
    '        {{\n'
    '            "category_id": "cc771f63-74e7-4c61-882e-3eda9d8ba5d7",\n'
    '            "category_name": "Price"\n'
    '        }}\n'
    '    ],\n'
    '    "classification_instructions": []\n'
    '}}\n'
)

_samples_ai_message_template_1 = (
    '{{\n'
    '    "keywords": [\n'
    '        "bad service",\n'
    '        "slow",\n'
    '        "food",\n'
    '        "tip",\n'
    '        "terrible",\n'
    '        "waitresses"\n'
    '    ],\n'
    '    "category_id": "f6ff5bc3-aca0-4e4a-8627-e760d0aca78f",\n'
    '    "category_name": "Experience"\n'
    '}}\n'
)

_human_message_template_0 = (
    '{{\n'
    '    "input_text": {query_json_str},\n'
    '    "categories": {categories_json_str},\n'
    '    "classification_instructions": {instruction_json_str}\n'
    '}}\n'
)

_human_message_template_1 = (
    '{query}'
)

PROMPT_TEMPLATE_CLASSIFY_INTENT = ChatPromptTemplate.from_messages(
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