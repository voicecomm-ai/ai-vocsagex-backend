from typing import Dict, List
import re
import json
import inspect

from langchain_core.messages.base import BaseMessage
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    ToolCall,
    UsageMetadata, 
)
from langchain.tools import BaseTool

def filter_valid_args(cls, params: Dict) -> Dict:
    '''
    将继承自 Pydantic v2 BaseModel 的子类中未声明的字段从参数字典中剔除。
    '''

    if not hasattr(cls, "model_fields"):
        return {}

    allowed = set(cls.model_fields.keys())
    # 支持 alias
    for field in cls.model_fields.values():
        if field.alias:
            allowed.add(field.alias)

    return {k: v for k, v in params.items() if k in allowed}

def usage_to_output(usage: UsageMetadata) -> Dict:
    return {
        "prompt_tokens": usage["input_tokens"], 
        "completion_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
    }

def generate_messages_from_dict(inputs: List[Dict]) -> List[BaseMessage]:
    out = []
    for item in inputs:
        role = item.get('role')
        text = item.get('content')
        images = item.get('images', [])

        content = [
            {
                "type": "text", 
                "text": text, 
            }
        ]

        if images:
            images_content = [
                {
                    "type": "image_url", 
                    "image_url": image
                }
                for image in images
            ]
            content.extend(images_content)
            
        if role == 'user' or role == 'human':
            out.append(HumanMessage(content=content))
        elif role == 'system':
            out.append(SystemMessage(content=content))
        elif role == 'assistant' or role == 'ai':
            out.append(AIMessage(content=content))
        else:
            raise ValueError(f'Unsupported message type - "{role}".')
    return out

def convert_message_to_str(msg: BaseMessage) -> str:
    if isinstance(msg, HumanMessage):
        role = 'User'
    elif isinstance(msg, AIMessage):
        role = 'Assistant'
    elif isinstance(msg, SystemMessage):
        role = 'System'
    elif isinstance(msg, ToolMessage):
        role = 'Tool'
    else:
        raise ValueError(f'Unsupported message type: {type(msg)}')
    
    return f'{role}: {msg.content}'

def generate_tool_parameters_ollama(schema: Dict) -> Dict:
    parameters = {
        'type': 'object',
        'properties': {},
        'required': schema.get('required', [])
    }

    for parameter, details in schema.get('properties', {}).items():
        parameters['properties'][parameter] = {}
        if 'type' in details.keys() and details['type'].startswith('array'):
            parameters['properties'][parameter]['type'] = 'array'
            parameters['properties'][parameter]['items'] = {'type': details['items']['type']}
        else:
            parameters['properties'][parameter]['type'] = details['type']

        parameters['properties'][parameter]['description'] = details['description']

    return parameters

def convert_tool_to_dict_ollama(tool: BaseTool) -> Dict:
    tool_item = {
        'type': 'function',
        'function': {
            'name': tool.name,
            'description': tool.description or "",
            'parameters': generate_tool_parameters_ollama(tool.args_schema.model_json_schema())
        }
    }

    return tool_item

def convert_chat_to_aimessage_ollama(msg: Dict) -> AIMessage:
    content = msg.get('content', '')

    tool_calls_list = msg.get('tool_calls', [])
    tool_calls: List[ToolCall] = []

    for call in tool_calls_list:
        function = call.get("function", {})
        name = function.get("name")
        arguments = function.get("arguments", {})

        # 如果 arguments 是字符串（有些接口这样返回），尝试解析
        if isinstance(arguments, str):
            import json
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        tool_calls.append(ToolCall(name=name, args=arguments, id=None))

    return AIMessage(content=content, tool_calls=tool_calls)

def convert_completion_to_aimessage_ollama(msg: str) -> AIMessage:
    return AIMessage(content=msg)

def remove_think_tag(text: str) -> str:
    text = re.sub(r'<think>.*?</think>\s*\n{0,2}', '', text, flags=re.DOTALL)
    text = re.sub(r'.*?</think>\s*\n{0,2}', '', text, flags=re.DOTALL)
    return text



def extract_react_final_answer(text: str) -> str:
    """
    从ReAct模式输出中提取Final Answer部分（支持多行内容）
    
    Args:
        text: ReAct模式的完整输出文本
        
    Returns:
        str: 提取的Final Answer内容，如果未找到则返回原始文本
    """
    pattern = r'Final\s+Answer\s*:\s*(.+?)(?=\n\s*(?:Thought|Action|Observation|Final Answer)\s*:|$)'
    matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
    
    if matches:
        final_answer = matches[-1].group(1).strip()
        final_answer = re.sub(r'\n\s*Observation\s*:.*$', '', final_answer, flags=re.DOTALL | re.IGNORECASE)
        final_answer = re.sub(r'\n\s*Thought\s*:.*$', '', final_answer, flags=re.DOTALL | re.IGNORECASE)
        final_answer = re.sub(r'\n\s*Action\s*:.*$', '', final_answer, flags=re.DOTALL | re.IGNORECASE)
        
        final_answer = re.sub(r'</?(?:think|thinking|reasoning|thought)(?:\s[^>]*)?>',
                               '', final_answer, flags=re.IGNORECASE)
        
        final_answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', final_answer).strip()
        
        return final_answer
    else:

        kb_no_result_patterns = [
            'No relevant information found in the knowledge base',
            'There is no relevant information in the knowledge base',
            '知识库中没有相关信息',
            '未找到相关信息'
        ]
        has_kb_no_result = any(pattern in text for pattern in kb_no_result_patterns)
        
        if has_kb_no_result:
    
            first_action_pos = next((m.start() for m in re.finditer(r'\bAction\s*:', text, re.IGNORECASE)), len(text))
            first_thought_section = text[:first_action_pos]
            
           
            first_thought_pattern = r'Thought\s*:\s*(.+?)(?=\n\s*(?:Thought|Action|Observation|Final Answer)\s*:|$)'
            first_thought_match = re.search(first_thought_pattern, first_thought_section, re.IGNORECASE | re.DOTALL)
            
            if first_thought_match:
                first_thought = first_thought_match.group(1).strip()
                
                if len(first_thought) > 50 and not first_thought.lower().startswith('error'):
                    
                    first_thought = re.sub(r'</?(?:think|thinking|reasoning|thought)(?:\s[^>]*)?>',
                                          '', first_thought, flags=re.IGNORECASE)
                    
                    first_thought = re.sub(r'\n\s*\n\s*\n+', '\n\n', first_thought).strip()
                    return first_thought
        
        # 策略1: 检查是否有工具调用尝试但没有实际执行（如调用不存在的工具）
        action_matches = list(re.finditer(r'\bAction\s*:', text, re.IGNORECASE))
        observation_matches = list(re.finditer(r'\bObservation\s*:', text, re.IGNORECASE))
        
        # 如果有 Action 但没有真实的 Observation（或只有幻觉的 Observation 被截断）
        if action_matches and len(observation_matches) == 0:
           
            first_action_pos = action_matches[0].start()
            first_thought_section = text[:first_action_pos]
            
            first_thought_pattern = r'Thought\s*:\s*(.+?)(?=\n\s*(?:Thought|Action|Observation|Final Answer)\s*:|$)'
            first_thought_match = re.search(first_thought_pattern, first_thought_section, re.IGNORECASE | re.DOTALL)
            
            if first_thought_match:
                first_thought = first_thought_match.group(1).strip()
                
                if len(first_thought) > 50 and not first_thought.lower().startswith('error'):
                    
                    first_thought = re.sub(r'</?(?:think|thinking|reasoning|thought)(?:\s[^>]*)?>',
                                          '', first_thought, flags=re.IGNORECASE)
                    
                    first_thought = re.sub(r'\n\s*\n\s*\n+', '\n\n', first_thought).strip()
                    return first_thought
        
        # 策略1.5: 检查是否有幻觉的 Observation（有 Observation 但没有 Action）
        if len(observation_matches) > 0 and len(action_matches) == 0:
            
            first_obs_pos = observation_matches[0].start()
            first_thought_section = text[:first_obs_pos]
            
            first_thought_pattern = r'Thought\s*:\s*(.+?)(?=\n\s*(?:Thought|Action|Observation|Final Answer)\s*:|$)'
            first_thought_match = re.search(first_thought_pattern, first_thought_section, re.IGNORECASE | re.DOTALL)
            
            if first_thought_match:
                first_thought = first_thought_match.group(1).strip()
                
                if len(first_thought) > 30 and not first_thought.lower().startswith('error'):
                    
                    first_thought = re.sub(r'</?(?:think|thinking|reasoning|thought)(?:\s[^>]*)?>',
                                          '', first_thought, flags=re.IGNORECASE)
                    
                    first_thought = re.sub(r'\n\s*\n\s*\n+', '\n\n', first_thought).strip()
                    return first_thought
        
      
        last_action_or_obs = max(
            (m.end() for m in re.finditer(r'\b(?:Action|Observation)\s*:', text, re.IGNORECASE)),
            default=-1
        )
        
        if last_action_or_obs >= 0:
            
            remaining_text = text[last_action_or_obs:]
            
            
            has_action_or_obs = bool(re.search(r'\b(?:Action|Observation)\s*:', remaining_text, re.IGNORECASE))
            
            if not has_action_or_obs:
                
                thought_pattern = r'Thought\s*:\s*(.+?)(?=\n\s*(?:Thought|Action|Observation|Final Answer)\s*:|$)'
                thought_matches = list(re.finditer(thought_pattern, remaining_text, re.IGNORECASE | re.DOTALL))
                
                if thought_matches:
                    
                    thoughts = [m.group(1).strip() for m in thought_matches]
                    combined_thoughts = '\n\n'.join(thoughts)
                    
                    
                    if len(combined_thoughts) > 50 and not combined_thoughts.lower().startswith('error'):
                        
                        combined_thoughts = re.sub(r'</?(?:think|thinking|reasoning|thought)(?:\s[^>]*)?>',
                                                    '', combined_thoughts, flags=re.IGNORECASE)
                        
                        combined_thoughts = re.sub(r'\n\s*\n\s*\n+', '\n\n', combined_thoughts).strip()
                        return combined_thoughts
        
        # 策略3: 从 Observation 中提取有用信息
        
        observation_pattern = r'Observation\s*:\s*(.+?)(?=\n\s*(?:Thought|Action|Observation|Final Answer)\s*:|$)'
        observation_matches = list(re.finditer(observation_pattern, text, re.IGNORECASE | re.DOTALL))
        
        if observation_matches:
            
            last_observation = observation_matches[-1].group(1).strip()
            
            
            if not (last_observation.lower().startswith('error') or 'exception' in last_observation.lower()):
                
                cleaned_obs = re.sub(r'\[Content truncated.*?\]', '', last_observation, flags=re.IGNORECASE)
                cleaned_obs = cleaned_obs.strip()
                
                
                kb_indicators = [
                    '发展规划', '未来展望', '技术研发', '区位优势', '政策支持',
                    '产业服务', '产业基地', '产业园区', '科技园', '技术实力',
                    'AI智能产业', '人工智能产业', '声通科技',
                    'development plan', 'future outlook', 'technical', 'location advantage'
                ]
                
                has_kb_indicators = sum(1 for indicator in kb_indicators if indicator in cleaned_obs) >= 1
                
                is_likely_kb_result = (
                    (len(cleaned_obs) > 500 and has_kb_indicators) or  
                    (has_kb_indicators and len(cleaned_obs) < 100)  
                )
                
                
                if is_likely_kb_result:
                    
                    obs_position = observation_matches[-1].end()
                    remaining_after_obs = text[obs_position:].strip()
                    has_follow_up_thought = bool(re.search(r'\bThought\s*:', remaining_after_obs, re.IGNORECASE))
                    
                    
                    if not has_follow_up_thought:
                        
                        first_action_match = re.search(r'\bAction\s*:', text, re.IGNORECASE)
                        if first_action_match:
                            first_thought_section = text[:first_action_match.start()]
                            first_thought_pattern = r'Thought\s*:\s*(.+?)(?=\n\s*(?:Thought|Action|Observation|Final Answer)\s*:|$)'
                            first_thought_match = re.search(first_thought_pattern, first_thought_section, re.IGNORECASE | re.DOTALL)
                            
                            if first_thought_match:
                                first_thought = first_thought_match.group(1).strip()
                                
                                first_thought_cleaned = re.sub(r'</?(?:think|thinking|reasoning|thought)(?:\s[^>]*)?>',
                                                              '', first_thought, flags=re.IGNORECASE)
                                
                                first_thought_cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', first_thought_cleaned).strip()
                                
                                
                                meta_thinking_patterns = [
                                    '需要', '应该', '可以使用', '调用', '查询', '获取', '工具',
                                    'need', 'should', 'can use', 'call', 'query', 'retrieve', 'tool'
                                ]
                                is_meta_thinking = any(pattern in first_thought_cleaned.lower() for pattern in meta_thinking_patterns)
                                
                                
                                if len(first_thought_cleaned) > 50 and not first_thought_cleaned.lower().startswith('error') and not is_meta_thinking:
                                    return first_thought_cleaned
                        
                        
                        return "抱歉，知识库中未找到与您查询相关的信息。请尝试换个方式提问，或者我可以使用我的知识为您解答。"
                
                
                if len(cleaned_obs) > 5:
                    
                    data_keywords = [
                        
                        '车次', 'train', '票', 'ticket', '出发', 'depart', '到达', 'arrive',
                        
                        '城市', 'city', '气温', 'temperature', '天气', 'weather', '湿度', 'humidity', '风速', 'wind',
                        
                        '时间', 'time', '日期', 'date',
                        
                        ':', '：', '{', '}'
                    ]
                    
                    
                    has_structure = any(keyword in cleaned_obs for keyword in data_keywords)
                    
                    
                    is_datetime = bool(re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', cleaned_obs))
                    
                    if has_structure or is_datetime:
                        
                        if cleaned_obs.startswith('{') and cleaned_obs.endswith('}'):
                            try:
                                data = json.loads(cleaned_obs)
                                
                                formatted = "\n".join([f"- {k}: {v}" for k, v in data.items()])
                                return f"根据查询结果：\n\n{formatted}"
                            except:
                                return f"根据查询结果：\n\n{cleaned_obs}"
                        else:
                            return f"根据查询结果：\n\n{cleaned_obs}"
        
        # 策略4: 如果上述方法都失败，尝试移除所有 ReAct 格式标记进行通用清理
        cleaned = re.sub(r'^(?:Thought|Action|Observation)\s*:.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
       
        cleaned = re.sub(r'</?(?:think|thinking|reasoning|thought)(?:\s[^>]*)?>',
                        '', cleaned, flags=re.IGNORECASE)
        
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned).strip()
        
        
        if not cleaned or len(cleaned) < 10:
            return ""
        
        
        if 'Action Input:' in cleaned or 'Action Input' in cleaned:
            return ""
        
        
        meta_only_patterns = [
            '需要了解', '需要确认', '需要获取', '需要查询',
            'need to understand', 'need to confirm', 'need to get', 'need to query'
        ]
        if any(pattern in cleaned.lower() for pattern in meta_only_patterns) and len(cleaned) < 100:
            return ""
        
        return cleaned