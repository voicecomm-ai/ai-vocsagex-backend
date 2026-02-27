'''
    生成python3代码
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
    "You are an expert programmer. Your task is to generate python3 code based on the provided instructions.\n"
    "Inside <instructions></instructions> XML tags, there is a text representing the provided instructions.\n"
    "<instructions>\n"
    "{instruction}\n"
    "</instructions>\n"
    "Write the code in python3.\n"
    "Return the raw code directly.\n"
    "\n"
    "### Steps to Follow\n"
    "Step 1: Carefully read and understand the provided instructions.\n"
    "Step 2: Identify the expected input and output, including data types and formats.\n"
    "Step 3: Design a logical plan or algorithm to solve the problem efficiently.\n"
    "Step 4: Write clean and readable code that follows the plan, using appropriate variable names.\n"
    "\n"
    "### Constraint\n"
    "1. You need to define a function named 'main' and all your code needs to be implemented in the 'main' function.\n"
    "2. You can modify the arguments of the 'main' function, but include appropriate type hints.\n"
    "3. The 'main' function must return a dictionary(dict), e.g., `{{\"output_name\": output_value}}`.\n"
    "4. The returned dictionary should contain at least one key-value pair.\n"
    "5. You may ONLY use the following libraries in your code:\n"
    "   - json\n"
    "   - datetime\n"
    "   - math\n"
    "   - random\n"
    "   - re\n"
    "   - string\n"
    "   - sys\n"
    "   - time\n"
    "   - traceback\n"
    "   - uuid\n"
    "   - os\n"
    "   - base64\n"
    "   - hashlib\n"
    "   - hmac\n"
    "   - binascii\n"
    "   - collections\n"
    "   - functools\n"
    "   - operator\n"
    "   - itertools\n"
    "6. Provide ONLY the code without any additional explanations, comments, or markdown formatting.\n"
    "7. DO NOT use markdown code blocks (``` or ```python). Return the raw code directly.\n"
    "8. The code should start immediately after this instruction, without any preceding newlines or spaces.\n"
    "9. The code should be complete, functional, and follow best practices for python3.\n"
    "\n"
    "### Example\n"
    "def main(arg1: str, arg2: int) -> dict:\n"
    "    return {{\n"
    "        \"result\": arg1 * arg2,\n"
    "    }}\n"
    "\n"
    "\n"
    "Generated Code:\n"
    "/no_think"
)

PROMPT_TEMPLATE_GENERATE_PYTHON3_CODE = ChatPromptTemplate.from_messages(
    messages=[
        HumanMessagePromptTemplate.from_template(_human_message_template),
    ]
)
