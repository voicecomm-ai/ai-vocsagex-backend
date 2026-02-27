'''
    根据查询，对长期记忆进行准确的筛选
'''

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
)

'''
    Template variables:
      - query: 用户问题
      - memories_json_str: 记忆结构JSON字符串
'''

_system_message_template = (

)

_samples_human_message_template_0 = (

)

_samples_ai_message_template_0 = (

)

_samples_human_message_template_1 = (

)

_samples_ai_message_template_1 = (
    
)

_human_message_template = (

)

PROMPT_TEMPLATE_FILTER_MEMORY = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessagePromptTemplate.from_template(_system_message_template),
        HumanMessagePromptTemplate.from_template(_samples_human_message_template_0),
        AIMessagePromptTemplate.from_template(_samples_ai_message_template_0),
        HumanMessagePromptTemplate.from_template(_samples_human_message_template_1),
        AIMessagePromptTemplate.from_template(_samples_ai_message_template_1),
        HumanMessagePromptTemplate.from_template(_human_message_template),
    ]
)