
# 系统提示词内的工具信息模板
TOOL_MESSAGES_TEMPLATE = '''## Available Tools 
Understand the capabilities of the tool through its description.
Use tool when you think it necessary.

The tools are as follows:
{%- for name, description in tools %}
- {{ name }}: {{ description }}
{%- endfor %}
'''

# 长期记忆模板
MEMORY_MESSAGE_TEMPLATE = '''## Long-term Memory
Below is the user’s long-term stable information. 
Use it only when relevant, and do not mention it proactively.

The contents are as follows:
{%- for mem in memories %}
- {{ mem }}
{%- endfor %}
'''

# 聊天历史摘要模板
HISTORY_SUMMARY_MESSAGE_TEMPLATE = '''## Chat History Compressed Summary
Below is a compressed summary of the earlier chat history, used to maintain context continuity.

The contents are as follows:
{{ history_summary }}
'''

# 聊天摘要提取的系统提示词
HISTORY_SUMMARY_SYSTEM_PROMPT = '''You are a conversation context compressor for an AI assistant.

Your task is NOT to write a narrative summary.
Your task is to extract and preserve factual context from earlier conversation turns
that MUST remain valid and be respected in subsequent responses.

The output will be used as system-level context in the next interaction.

Rules:
- Focus on information that directly affects how the assistant should respond next.
- Explicitly retain:
  - User corrections, negations, or clarifications.
  - Constraints, requirements, or prohibitions that are still in effect.
  - Decisions or agreements that have been reached.
  - Open or unresolved questions.
- Do NOT merge unrelated points.
- Do NOT soften or rephrase constraints.
- Do NOT add any new inferences, assumptions, or suggestions.
- Do NOT include greetings, politeness, or filler content.
- If a category has no relevant content, omit it.
- Use concise bullet points, not prose paragraphs.
- Use the third person.
- The language must match the main language of the conversation.
- Output ONLY the compressed context text, with no explanations.
'''

# 聊天摘要提取的用户查询
HISTORY_SUMMARY_HUMAN_PROMPT = '''Compress the conversation into a structured context record
that preserves all information still relevant for the next response.
'''
