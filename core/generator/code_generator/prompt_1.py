'''
    生成javascript代码
'''

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

'''
    Template variables:
      - instruction: 用户指令
'''
_human_message_template = (
    "### Task\n"
    "You are an expert programmer. Your task is to generate javascript code based on the provided instructions.\n"
    "Inside <instructions></instructions> XML tags, there is a text representing the provided instructions.\n"
    "<instructions>\n"
    "{instruction}\n"
    "</instructions>\n"
    "Write the code in javascript.\n"
    "Return the raw code directly.\n"
    "\n"
    "### Steps to Follow\n"
    "Step 1: Carefully read and understand the provided instructions.\n"
    "Step 2: Identify the expected input and output.\n"
    "Step 3: Design a logical plan or algorithm to solve the problem efficiently.\n"
    "Step 4: Write clean and readable code that follows the plan, using appropriate variable names.\n"
    "\n"
    "### Constraint\n"
    "1. You need to define a function named 'main' and all your code needs to be implemented in the 'main' function.\n"
    "2. You can modify the arguments of the 'main' function, but include appropriate JSDoc annotations.\n"
    "3. The 'main' function must return a object, e.g., `{{output_name: output_value}}`.\n"
    "4. The returned dictionary should contain at least one key-value pair.\n"
    "5. Provide ONLY the code without any additional explanations, comments, or markdown formatting.\n"
    "6. DO NOT use markdown code blocks (``` or ```javascript). Return the raw code directly.\n"
    "7. The code should start immediately after this instruction, without any preceding newlines or spaces.\n"
    "8. The code should be complete, functional, and follow best practices for javascript.\n"
    "\n"
    "### Example\n"
    "function main(arg1, arg2) {{\n"
    "    return {{\n"
    "        result: arg1 * arg2\n"
    "    }};\n"
    "}}\n"
    "\n"
    "\n"
    "Generated Code:\n"
    "/no_think"
)

PROMPT_TEMPLATE_GENERATE_JAVASCRIPT_CODE = ChatPromptTemplate.from_messages(
    messages=[
        HumanMessagePromptTemplate.from_template(_human_message_template),
    ]
)
