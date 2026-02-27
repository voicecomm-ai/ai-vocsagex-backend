'''
    FunctionCall 提示词
'''

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate, 
)

from langchain_core.messages import (
    BaseMessage, 
    HumanMessage, 
    SystemMessage, 
    AIMessage, 
    ToolMessage, 
    FunctionMessage, 
    ToolCall,
)

'''
    Template variables:
      - instruction: 提取时的指令
      - query: 用户查询
      - args_schema_json: 结构json字符串
'''

_system_message_template = (
    "### Task\n"
    "You are an assistant dedicated to extracting structured parameters from user-provided information "
    "and returning them strictly through a function call.\n"
    "\n"
    "### Instruction\n"
    "The user may provide additional extraction instructions to guide you (e.g., which fields to focus on, formatting requirements, filtering rules).\n"
    "- Always read and apply the user-provided instruction when extracting parameters.\n"
    "- If an instruction conflicts with default behavior, the instruction takes priority.\n"
    "- If no instruction is provided, follow the default extraction steps.\n"
    "\n"
    "### Steps to Follow\n"
    "1. Read the predefined schema of the function `extract_parameters`.\n"
    "2. Always scan the input text to find values for the schema fields (JSON keys, key-value pairs, or natural language mentions), even if the input is extremely long (10,000+ characters).\n"
    "3. Normalize values to the required data types and formats (e.g., integer, float, boolean, ISO date).\n"
    "4. If multiple values exist, select the most explicit one; if uncertain, set the field to null.\n"
    "5. Apply any user-provided extraction instruction if available.\n"
    "6. Do not include explanations, summaries, reasoning, or <think> tags.\n"
    "7. Always return the result strictly as a function call to `extract_parameters`.\n"
    "\n"
    "### Constraint\n"
    "- Do not generate any content in the assistant message.\n"
    "- Do not summarize or rephrase the input, regardless of length.\n"
    "- Do not fabricate or guess values; use null if unavailable.\n"
    "- The output must be a valid tool call with `name=\"extract_parameters\"` and properly formatted `args`.\n"
    "\n"
    "/no_think"
)

_human_message_template = (
    "### Instruction\n"
    "- Only extract the fields according to the schema.\n"
    "- Ignore all other information, do not summarize.\n"
    "- {instruction}\n"
    "\n"
    "### Input\n"
    "{query}\n"
)

PROMPT_TEMPLATE_EXTRACT_PARAMETER = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessagePromptTemplate.from_template(_system_message_template),
        HumanMessage(content="I want to go to Suzhou for a two-day trip."),
        AIMessage(content="", tool_calls=[ToolCall(name="extract_parameters", args={"city": "Suzhou", "days": 2}, id=None)]),
        ToolMessage(content="Great! You have called the function with the correct parameters.", tool_call_id = "1234"),
        HumanMessage(content="Search for knowledge about AI."),
        AIMessage(content="", tool_calls=[ToolCall(name="extract_parameters", args={"query": "knowledge about AI"}, id=None)]),
        ToolMessage(content="Great! You have called the function with the correct parameters.", tool_call_id = "5678"),
        HumanMessagePromptTemplate.from_template(_human_message_template),
    ]
)