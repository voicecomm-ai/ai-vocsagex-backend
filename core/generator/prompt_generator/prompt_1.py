'''
    从生成的提示词中提取变量
'''

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

'''
    Template variables:
      - prompt: 生成的提示词
'''
_human_message_template = (
    "### Task\n"
    "I need to extract the following information from the input text. The <information to be extracted> tag specifies the 'type', 'description' and 'required' of the information to be extracted. \n"
    "<information to be extracted>\n"
    "variables name bounded two double curly brackets. Variable name has to be composed of number, english alphabets and underline and nothing else. \n"
    "</information to be extracted>\n"
    "\n"
    "### Steps to Follow\n"
    "Step 1: Carefully read the input and understand the structure of the expected output.\n"
    "Step 2: Extract relevant parameters from the provided text based on the name and description of object. \n"
    "Step 3: Structure the extracted parameters to JSON object as specified in <structure>.\n"
    "Step 4: Ensure that the list of variable_names is properly formatted and valid. The output should not contain any XML tags. Output an empty list if there is no valid variable name in input text. \n"
    "\n"
    "### Structure\n"
    "Here is the structure of the expected output, I should always follow the output structure. \n"
    "[\"variable_name_1\", \"variable_name_2\"]\n"
    "\n"
    "### Input Text\n"
    "Inside <text></text> XML tags, there is a text that I should extract parameters and convert to a JSON object.\n"
    "<text>\n"
    "{prompt}\n"
    "</text>\n"
    "\n"
    "### Constraint\n"
    "- You should always output a valid list. Output nothing other than the list of variable_name. Output an empty list if there is no variable name in input text.\n"
    "/no_think"
)

PROMPT_TEMPLATE_EXTRACT_VARIABLES = ChatPromptTemplate.from_messages(
    messages=[
        HumanMessagePromptTemplate.from_template(_human_message_template),
    ]
)
