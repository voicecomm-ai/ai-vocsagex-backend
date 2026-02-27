'''
    查询重写
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
'''

_system_message_template = (
    "You are a query rewriting assistant.\n"
    "Your task is to transform a user's natural language query into a concise, retrieval-oriented form\n"
    "that can be used to search a vector database.\n"
    "\n"
    "### Requirements:\n"
    "- Keep only the essential meaning of the query.\n"
    "- Remove greetings, filler words, or irrelevant context.\n"
    "- Expand ambiguous references (e.g., \"he\", \"yesterday\") if context is available.\n"
    "- Use clear and descriptive keywords that capture the core intent.\n"
    "- Do not answer the query, only rewrite it for retrieval.\n"
    "- If the query should be ignored (e.g., greetings like \"hello\", \"hi\"), set the query field to an empty string.\n"
    "\n"
    "### Output Format:\n"
    "You must output a JSON object with the following structure:\n"
    "{{\n"
    "  \"query\": \"<rewritten query string or empty string if should be ignored>\"\n"
    "}}\n"
    "\n"
    "### Constraint:\n"
    "1. Provide ONLY the JSON object without any additional explanations, comments, or markdown formatting.\n"
    "2. DO NOT use markdown code blocks (``` or ```json). Return the raw JSON directly.\n"
    "\n"
    "### Example:\n"
    "Input: 我昨天买的电脑是什么？\n"
    "Output:\n"
    '{{\n'
    '  "query": "用户昨天购买的电脑型号"\n'
    '}}'
)

_samples_human_message_template_0 = (
    "Inside <query></query> XML tags, there is the original user's natural language query.\n"
    "<query>\n"
    "我昨天买的电脑是什么？\n"
    "</query>\n"
)

_samples_ai_message_template_0 = (
    '{{\n'
    '  "query": "用户昨天购买的电脑型号"\n'
    '}}'
)

_samples_human_message_template_1 = (
    "Inside <query></query> XML tags, there is the original user's natural language query.\n"
    "<query>\n"
    "记得我喜欢喝什么吗？\n"
    "</query>\n"
)

_samples_ai_message_template_1 = (
    '{{\n'
    '  "query": "用户喜欢的饮品"\n'
    '}}'
)

_samples_human_message_template_2 = (
    "Inside <query></query> XML tags, there is the original user's natural language query.\n"
    "<query>\n"
    "你好，早上好！\n"
    "</query>\n"
)

_samples_ai_message_template_2 = (
    '{{\n'
    '  "query": ""\n'
    '}}'
)

_human_message_template = (
    "Inside <query></query> XML tags, there is the original user's natural language query.\n"
    "<query>\n"
    "{query}\n"
    "</query>\n"
)

PROMPT_TEMPLATE_REWRITE_QUERY = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessagePromptTemplate.from_template(_system_message_template),
        HumanMessagePromptTemplate.from_template(_human_message_template),
    ]
)