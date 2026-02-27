'''
    根据提示词，生成会话标题
'''

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

'''
    Template variables:
      - query: 用户查询
'''
_human_message_template = (
    "### Task\n"
    "Generate a concise conversation title based on the user's query.\n"
    "\n"
    "### Steps to Follow\n"
    "1. Identify the core intent and key topic of the user's query.\n"
    "2. Determine the primary language used in the query.\n"
    "3. Generate a short title that accurately summarizes the query.\n"
    "4. Ensure the title uses the same language as the query.\n"
    "5. Refine the title to be clear, neutral, and concise.\n"
    "\n"
    "### Example\n"
    "Input: Query: 如何在 LangGraph 中实现多知识库 RAG\n"
    "Output: LangGraph 多知识库 RAG 实现\n"
    "\n"
    "Input: Query: How to use LangChain with structured output\n"
    "Output: LangChain Structured Output Usage\n"
    "\n"
    "### Input\n"
    "Query: {query}\n"
    "\n"
    "### Constraints\n"
    "- The title must be written in the same language as the query.\n"
    "- Do NOT translate the query into another language.\n"
    "- Do NOT introduce information not present in the query.\n"
    "- Do NOT use marketing, emotional, or exaggerated wording.\n"
    "- The title should be concise:\n"
    "  - No more than 12 Chinese characters, OR\n"
    "  - No more than 8 English words.\n"
    "- Output ONLY the title text, with no explanations or additional content.\n"
    "/no_think"
)

PROMPT_TEMPLATE_GENERATE_CONVERSATION_TITLE = ChatPromptTemplate.from_messages(
    messages=[
        HumanMessagePromptTemplate.from_template(_human_message_template),
    ]
)
