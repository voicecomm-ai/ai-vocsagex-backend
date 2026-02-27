'''
    Prompt 提示词
'''

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate, 
)

'''
    Template variables:
      - instruction: 提取时的指令
      - query: 用户查询
      - args_schema_json: 结构json字符串
'''

_system_message_template = (
    "You are a helpful assistant tasked with extracting structured information based on specific criteria provided. Follow the guidelines below to ensure consistency and accuracy.\n"
    "### Task\n"
    "Always call the `extract_parameters` function with the correct parameters. Ensure that the information extraction is contextual and aligns with the provided criteria.\n"
    "\n"
    "### Instructions:\n"
    "Some additional information is provided below. Always adhere to these instructions as closely as possible:\n"
    "<instruction>\n"
    "{instruction}\n"
    "</instruction>\n"
    "Steps:\n"
    "1. Extract the relevant information based on the criteria given, output multiple values if there is multiple relevant information that match the criteria in the given text. \n"
    "2. When extracting parameters, if a description exists in the JSON schema, use it as the primary reference for extraction; if no description is provided, use the field name as the reference.\n"
    "3. Generate a well-formatted output using the defined functions and arguments.\n"
    "4. Use the `extract_parameter` function to create structured outputs with appropriate parameters.\n"
    "5. Do not include any XML tags in your output.\n"
    "\n"
    "### Example\n"
    "To illustrate, if the task involves extracting a user's name and their request, your function call might look like this: Ensure your output follows a similar structure to examples.\n"
    "\n"
    "### Final Output\n"
    "Produce well-formatted function calls in json without XML tags, as shown in the example.\n"
    "/no_think"
)

_samples_human_message_template_0 = (
    "### Structure\n"
    "Here is the structure of the JSON object, you should always follow the structure.\n"
    "<structure>\n"
    '{{"type": "object","properties": {{"city": {{"description": "游玩城市","type" : "string"}},"days": {{"description": "游玩天数","type" : "number"}}}},"required":["city", "days"]}}\n'
    "</structure>\n"
    "\n"
    "### Text to be converted to JSON\n"
    "Inside <text></text> XML tags, there is a text that you should convert to a JSON object.\n"
    "<text>\n"
    "I want to go to Suzhou for a two-day trip.\n"
    "</text>\n"
)

_samples_ai_message_template_0 = (
    '{{"city": "Suzhou", "days": 2}}'
)

_samples_human_message_template_1 = (
    "### Structure\n"
    "Here is the structure of the JSON object, you should always follow the structure.\n"
    "<structure>\n"
    '{{"type": "object","properties": {{"query": {{"description": "查询内容","type" : "string"}}}},"required":["query"]}}\n'
    "</structure>\n"
    "\n"
    "### Text to be converted to JSON\n"
    "Inside <text></text> XML tags, there is a text that you should convert to a JSON object.\n"
    "<text>\n"
    "Search for knowledge about AI.\n"
    "</text>\n"
)

_samples_ai_message_template_1 = (
    '{{"query": "knowledge about AI"}}'
)

_human_message_template = (
    "### Input\n"
    "<text>\n"
    "{query}\n"
    "</text>\n"
    "\n"
    "### Structure\n"
    "To extract parameters from the input, you should always follow the structure.\n"
    "<structure>\n"
    "{args_schema_json}\n"
    "</structure>\n"
    "\n"
    "Just output the raw json string without any markdown tags.\n"
    "/no_think"
    "### Output\n"
)

PROMPT_TEMPLATE_EXTRACT_PARAMETER = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessagePromptTemplate.from_template(_system_message_template),
        HumanMessagePromptTemplate.from_template(_samples_human_message_template_0),
        AIMessagePromptTemplate.from_template(_samples_ai_message_template_0),
        HumanMessagePromptTemplate.from_template(_samples_human_message_template_1),
        AIMessagePromptTemplate.from_template(_samples_ai_message_template_1),
        HumanMessagePromptTemplate.from_template(_human_message_template),
    ]
)