
_demo_prompt = (
    "### Task\n"
    ""
    "\n"
    "### Steps to Follow\n"
    "Step 1: \n"
    "Step 2: \n"
    "\n"
    "### Example\n"
    ""
    "\n"
    "### Constraint\n"
    "- "
    "- "
    "\n"
    "\n"
    "Output:\n"
)

# 如果需要关闭思考模式，在提示词末尾添加 /no_think

_output_dict = {
    'data': {       # dict or None
        
    },
    'usage': {      # dict or None
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
    }
}