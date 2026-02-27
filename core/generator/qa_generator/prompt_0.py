'''
    根据切分的文本片段，生成QA对
'''

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

'''
    Template variables:
      - language: QA语言
      - page_content: 文本片段
'''

_system_message_template = (
    "### Task\n"
    "The user will send a long text. Generate Question and Answer pairs only using the knowledge in the long text.\n"
    "\n"
    "### Steps to Follow\n"
    "Step 1: Understand and summarize the main content of this text.\n"
    "Step 2: What key information or concepts are mentioned in this text?\n"
    "Step 3: Decompose complex ideas or combine related points to form meaningful questions.\n"
    "Step 4: Generate questions and answers based on these key information and concepts.\n"
    "\n"
    "### Constraint\n"
    "- The questions should be clear and detailed, and the answers should be detailed and complete.\n"
    "- You must answer in {language}, in a style that is clear and detailed in {language}.\n"
    "- No language other than {language} should be used.\n"
    "- Your response needs to use the following format: Q1:\nA1:\nQ2:\nA2:...\n"
    "- Do NOT generate questions or answers about special characters, digit sequences, individual symbols, encoding issues, text formatting, or language detection.\n"
    "- Avoid questions that simply list characters, symbols, or guess the meaning of corrupted or unreadable text.\n"
    "- Do NOT speculate on technical or meta aspects such as encoding errors or text corruption.\n"
    "- Ignore meta-questions about the text's format, structure, or language.\n"
    "- Focus solely on extracting, combining, or inferring knowledge and concepts explicitly or implicitly expressed in the text.\n"
    "\n"
)

_human_message_template = (
    "{page_content}\n"
    "/no_think"
)

PROMPT_TEMPLATE_GENERATE_QA = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessagePromptTemplate.from_template(_system_message_template),
        HumanMessagePromptTemplate.from_template(_human_message_template),
    ]
)
