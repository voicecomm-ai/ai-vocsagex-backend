'''
    根据自然语言，生成JSON Schema
'''

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

'''
    Template variables:
      - description: JSON Schema 描述
'''

_system_message_template = (
    "### Task\n"
    "Your task is to convert simple user descriptions into properly formatted JSON Schema definitions.\n"
    "When a user describes data fields they need, generate a complete, valid JSON Schema that accurately represents those fields with appropriate types and requirements.\n"
    "\n"
    "### Instructions\n"
    "1. Analyze the user's description of their data needs.\n"
    "2. Identify each property that should be included in the schema.\n"
    "3. Determine the appropriate data type for each property.\n"
    "4. Decide which properties should be required.\n"
    "5. Generate a complete JSON Schema with proper syntax.\n"
    "6. Include appropriate constraints when specified (min/max values, patterns, formats).\n"
    "7. Provide ONLY the JSON Schema without any additional explanations, comments, or markdown formatting.\n"
    "8. DO NOT use markdown code blocks (``` or ``` json). Return the raw JSON Schema directly.\n"
    "\n"
    "### Examples\n"
    "#### Example 1\n"
    "**User Input:** I need name and age.\n"
    "**JSON Schema Output:**\n"
    '{{\n'
    '  "type": "object",\n'
    '  "properties": {{\n'
    '    "name": {{ "type": "string" }},\n'
    '    "age": {{ "type": "number" }}\n'
    '  }},\n'
    '  "required": ["name", "age"]\n'
    '}}\n'
    "\n"
    "#### Example 2\n"
    "**User Input:** I want to store information about books including title, author, publication year and optional page count.\n"
    "**JSON Schema Output:**\n"
    '{{\n'
    '  "type": "object",\n'
    '  "properties": {{\n'
    '    "title": {{ "type": "string" }},\n'
    '    "author": {{ "type": "string" }},\n'
    '    "publicationYear": {{ "type": "integer" }},\n'
    '    "pageCount": {{ "type": "integer" }}\n'
    '  }},\n'
    '  "required": ["title", "author", "publicationYear"]\n'
    '}}\n'
    "\n"
    "#### Example 3\n"
    "**User Input:** Create a schema for user profiles with email, password, and age (must be at least 18)\n"
    "**JSON Schema Output:**\n"
    '{{\n'
    '  "type": "object",\n'
    '  "properties": {{\n'
    '    "email": {{ \n'
    '      "type": "string",\n'
    '      "format": "email"\n'
    '    }},\n'
    '    "password": {{ \n'
    '      "type": "string",\n'
    '      "minLength": 8\n'
    '    }},\n'
    '    "age": {{ \n'
    '      "type": "integer",\n'
    '      "minimum": 18\n'
    '    }}\n'
    '  }},\n'
    '  "required": ["email", "password", "age"]\n'
    '}}\n'
    "\n"
    "#### Example 4\n"
    "**User Input:** I need album schema, the ablum has songs, and each song has name, duration, and artist.\n"
    "**JSON Schema Output:**\n"
    '{{\n'
    '    "type": "object",\n'
    '    "properties": {{\n'
    '        "properties": {{\n'
    '            "songs": {{\n'
    '                "type": "array",\n'
    '                "items": {{\n'
    '                    "type": "object",\n'
    '                    "properties": {{\n'
    '                        "name": {{\n'
    '                            "type": "string"\n'
    '                        }},\n'
    '                        "id": {{\n'
    '                            "type": "string"\n'
    '                        }},\n'
    '                        "duration": {{\n'
    '                            "type": "string"\n'
    '                        }},\n'
    '                        "aritst": {{\n'
    '                            "type": "string"\n'
    '                        }}\n'
    '                    }},\n'
    '                    "required": [\n'
    '                        "name",\n'
    '                        "id",\n'
    '                        "duration",\n'
    '                        "aritst"\n'
    '                    ]\n'
    '                }}\n'
    '            }}\n'
    '        }}\n'
    '    }},\n'
    '    "required": [\n'
    '        "songs"\n'
    '    ]\n'
    '}}\n'
    "\n"
    "\n"
    "Now, generate a JSON Schema based on my description.\n"
    "/no_think"
)

_human_message_template = (
    "{description}"
)

PROMPT_TEMPLATE_GENERATE_JSON_SCHEMA = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessagePromptTemplate.from_template(_system_message_template), 
        HumanMessagePromptTemplate.from_template(_human_message_template),
    ]
)