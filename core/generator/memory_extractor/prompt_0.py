'''
    根据问答，生成长期记忆
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
      - answer: LLM回答
      - existing_memory_list: 已有的记忆列表
'''

_system_message_template = (
    "### Task\n"
    "You are an assistant that extracts long-term memory from user interactions. \n"
    "Given the following user query and your answer, extract key information that should be remembered for future interactions with this specific user.\n"
    "If the content of a certain memory you extract is similar to that of an existing memory, skip it.\n"
    "\n"
    "### Rules\n"
    "1. Only extract meaningful facts, preferences, or important context that are relevant to this user.\n"
    "2. Ignore trivial, generic, or one-time information.\n"
    "3. Be concise and specific.\n"
    "4. For each distinct fact, preference, or piece of context, create a separate memory entry.\n"
    "5. Do not duplicate existing memories. If a potential memory is semantically similar to any entry in the existing memory list, skip it.\n"
    "6. If no meaningful memory can be extracted, return an empty array: [].\n"
    "7. Output all memories as a JSON array, where each element has:\n"
    '   - "memory_type": e.g., "preference", "fact", "context"\n'
    '   - "content": the information to remember (string)\n'
    '   - "source": "query", "answer", or "both"\n'
    '   - "user_related": bool  # indicates whether this memory is associated with the user\n'
    "\n"
    "### Constraint\n"
    "1. Provide ONLY the JSON array without any additional explanations, comments, or markdown formatting.\n"
    "2. DO NOT use markdown code blocks (``` or ``` json). Return the raw JSON Schema directly.\n"
    "3. The language of the element \"content\" must be consistent with the user query.\n"
    "\n"
    "### Example\n"
    "Input: 帮我记住，我喜欢红色；我喜欢在中午喝茶"
    "Output:\n"
    '[\n'
    '  {{\n'
    '    "memory_type": "preference",\n'
    '    "content": "用户喜欢红色。",\n'
    '    "source": "both",\n'
    '    "user_related": true\n'
    '  }},\n'
    '  {{\n'
    '    "memory_type": "preference",\n'
    '    "content": "用户喜欢在中午喝茶。",\n'
    '    "source": "both",\n'
    '    "user_related": true\n'
    '  }}\n'
    ']\n'
)

_samples_human_message_template_0 = (
    "### Query\n"
    "Inside <query></query> XML tags, there is the query in the interactions.\n"
    "<query>\n"
    "I love drinking coffee in the morning.\n"
    "</query>\n"
    "\n"
    "### Answer\n"
    "Inside <answer></answer> XML tags, there is the query in the interactions.\n"
    "<answer>\n"
    "Got it! You enjoy having coffee in the morning.\n"
    "</answer>\n"
    "\n"
    "### Existing Memory Contents"
    "Inside <contents></contents> XML tags, there is the existing memory contents.\n"
    "<contents>\n"
    "[\"User likes drinking coffee in the morning.\"]"
    "</contents>\n"
)

_samples_ai_message_template_0 = (
    '[]\n'
)

_samples_human_message_template_1 = (
    "### Query\n"
    "Inside <query></query> XML tags, there is the query in the interactions.\n"
    "<query>\n"
    "Hello, how are you?\n"
    "</query>\n"
    "\n"
    "### Answer\n"
    "Inside <answer></answer> XML tags, there is the query in the interactions.\n"
    "<answer>\n"
    "I'm fine, thank you!\n"
    "</answer>\n"
    "\n"
    "### Existing Memory Contents"
    "Inside <contents></contents> XML tags, there is the existing memory contents.\n"
    "<contents>\n"
    "[]"
    "</contents>\n"
)

_samples_ai_message_template_1 = (
    '[]\n'
)

_samples_human_message_template_2 = (
    "### Query\n"
    "Inside <query></query> XML tags, there is the query in the interactions.\n"
    "<query>\n"
    "帮我记住，我喜欢红色；我喜欢在中午喝茶\n"
    "</query>\n"
    "\n"
    "### Answer\n"
    "Inside <answer></answer> XML tags, there is the query in the interactions.\n"
    "<answer>\n"
    "已记住：你喜欢红色，并且喜欢在中午喝茶。\n"
    "</answer>\n"
    "\n"
    "### Existing Memory Contents"
    "Inside <contents></contents> XML tags, there is the existing memory contents.\n"
    "<contents>\n"
    "[]"
    "</contents>\n"
)

_samples_ai_message_template_2 = (
    '[\n'
    '  {{\n'
    '    "memory_type": "preference",\n'
    '    "content": "用户喜欢红色。",\n'
    '    "source": "both",\n'
    '    "user_related": true\n'
    '  }},\n'
    '  {{\n'
    '    "memory_type": "preference",\n'
    '    "content": "用户喜欢在中午喝茶。",\n'
    '    "source": "both",\n'
    '    "user_related": true\n'
    '  }}\n'
    ']\n'
)


_human_message_template = (
    "### Query\n"
    "Inside <query></query> XML tags, there is the query in the interactions.\n"
    "<query>\n"
    "{query}\n"
    "</query>\n"
    "\n"
    "### Answer\n"
    "Inside <answer></answer> XML tags, there is the query in the interactions.\n"
    "<answer>\n"
    "{answer}\n"
    "</answer>\n"
    "\n"
    "### Existing Memory Contents"
    "Inside <contents></contents> XML tags, there is the existing memory contents.\n"
    "<contents>\n"
    "{existing_memory_list}"
    "</contents>\n"
    "/no_think"
)

PROMPT_TEMPLATE_EXTRACT_MEMORY = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessagePromptTemplate.from_template(_system_message_template),
        HumanMessagePromptTemplate.from_template(_human_message_template),
    ]
)