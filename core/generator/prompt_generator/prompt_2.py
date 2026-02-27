'''
    根据提示词，生成开场白
'''

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

'''
    Template variables:
      - instruction: 用户指令
      - prompt: 生成的提示词
'''
_human_message_template = (
    "### Task\n"
    "Your task is to generate the opening statement of the agent based on the task description.\n"
    "\n"
    "### Steps to Follow\n"
    "Step 1: Identify the purpose of the chatbot from the task description and infer chatbot's tone  (e.g., friendly, professional, etc.) to add personality traits. \n"
    "Step 2: Determine the language of the answer based on the sentence \"{instruction}\".\n"
    "Step 3: Create a coherent and engaging opening statement.\n"
    "Step 4: Ensure the output is welcoming and clearly explains what the chatbot is designed to do. Do not include any XML tags in the output.\n"
    "\n"
    "### Example\n"
    "Example Input: \n"
    "Provide customer support for an e-commerce website\n"
    "Example Output: \n"
    "Welcome! I'm here to assist you with any questions or issues you might have with your shopping experience. Whether you're looking for product information, need help with your order, or have any other inquiries, feel free to ask. I'm friendly, helpful, and ready to support you in any way I can.\n"
    "\n"
    "### Input\n"
    "Inside <task_description></task_description> XML tags, there is a text representing the task description.\n"
    "<task_description>\n"
    "{prompt}\n"
    "</task_description>\n"
    "\n"
    "### Constraint\n"
    "- You just need to generate the output.\n"
    "- Please use the same language as the sentence \"{instruction}\".\n"
    "/no_think"
)

PROMPT_TEMPLATE_GENERATE_OPENING_STATEMENT = ChatPromptTemplate.from_messages(
    messages=[
        HumanMessagePromptTemplate.from_template(_human_message_template),
    ]
)
