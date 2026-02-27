from typing import (Sequence, List, Dict)
import copy
import json
import unicodedata

from langchain_core.messages.base import BaseMessage
from langchain_core.messages import (
    AIMessage, HumanMessage, ToolMessage, UsageMetadata, ToolCall
)

def get_history_turn(history: Sequence[BaseMessage]) -> int:
    '''
    获取聊天历史的对话轮次
    '''
    turn = 0
        
    if history:
        if not isinstance(history[0], HumanMessage):
            # 当不是以HumanMessage开始时，多记一轮
            turn = 1

        for msg in history:
            if isinstance(msg, HumanMessage):
                turn = turn + 1

    return turn

def truncate_history(history: Sequence[BaseMessage], depth: int) -> List[BaseMessage]:
    '''
    截断聊天历史，取后depth轮
    '''
    history_turns = []
    current_turn = []

    for msg in history:
        if isinstance(msg, HumanMessage):
            if current_turn:
                history_turns.append(current_turn)
                current_turn = []
        current_turn.append(msg)

    if current_turn:
        history_turns.append(current_turn)

    trimmed_history_turns = history_turns[-depth:]
    return [msg for turn in trimmed_history_turns for msg in turn]

def truncate_history_front(history: Sequence[BaseMessage], depth: int) -> List[BaseMessage]:
    '''
    截断聊天历史，取前depth轮
    '''
    history_turns = []
    current_turn = []

    for msg in history:
        if isinstance(msg, HumanMessage):
            if current_turn:
                history_turns.append(current_turn)
                current_turn = []
        current_turn.append(msg)

    if current_turn:
        history_turns.append(current_turn)

    trimmed_history_turns = history_turns[:depth]
    return [msg for turn in trimmed_history_turns for msg in turn]

def simplify_history(history: Sequence[BaseMessage]) -> List[BaseMessage]:
    '''
    简化聊天历史，仅保留AIMessage、HumanMessage、ToolMessage的重要字段
    '''
    new_history = []

    for msg in history:
        if isinstance(msg, AIMessage):
            ai_msg = AIMessage(
                content=msg.content,
                tool_calls=[
                    ToolCall(
                        name=tc["name"],
                        args=copy.deepcopy(tc["args"]),
                        id=None
                    ) for tc in msg.tool_calls
                ], 
            )

            if "reasoning_content" in msg.additional_kwargs:
                ai_msg.additional_kwargs["reasoning_content"] = msg.additional_kwargs["reasoning_content"]

            new_history.append(ai_msg)
        elif isinstance(msg, HumanMessage):
            new_history.append(HumanMessage(content=msg.content))
        elif isinstance(msg, ToolMessage):
            new_history.append(ToolMessage(
                content=msg.content, 
                name=getattr(msg, "name", None),
                tool_call_id="", 
                status=msg.status, 
            ))
        else:
            pass
    return new_history

def serialize_history(history: List[BaseMessage]) -> List[Dict]:
    outs = []
    for msg in history:
        if isinstance(msg, AIMessage):
            ai_obj = {
                "type": msg.type, 
                "content": msg.content, 
                "tool_calls": [
                    {
                        "type": tc["type"], 
                        "name": tc["name"], 
                        "args": copy.deepcopy(tc["args"]), 
                    }
                    for tc in msg.tool_calls
                ], 
                "additional_kwargs": {}, 
            }
            if "reasoning_content" in msg.additional_kwargs:
                ai_obj["additional_kwargs"]["reasoning_content"] = msg.additional_kwargs["reasoning_content"]

            outs.append(ai_obj)
        elif isinstance(msg, HumanMessage):
            outs.append(
                {
                    "type": msg.type, 
                    "content": msg.content, 
                }
            )
        elif isinstance(msg, ToolMessage):
            outs.append(
                {
                    "type": msg.type, 
                    "content": msg.content, 
                    "name": msg.name, 
                    "status": msg.status, 
                }
            )
        else:
            pass

    return outs

def deserialize_history(history: List[Dict]) -> List[BaseMessage]:
    outs = []
    for obj in history:
        obj_type = obj.get("type")
        if obj_type == "ai":
            outs.append(AIMessage(
                content=obj.get("content", ""), 
                tool_calls=[
                    ToolCall(
                        name=c.get("name", ""), 
                        args=c.get("args", {}), 
                        id=None, 
                    )
                    for c in obj.get("tool_calls", [])
                ],
                additional_kwargs=obj.get("additional_kwargs", {}), 
            ))
        elif obj_type == "human":
            outs.append(HumanMessage(
                content=obj.get("content", ""), 
            ))
        elif obj_type == "tool":
            outs.append(ToolMessage(
                content=obj.get("content", ""), 
                name=obj.get("name", ""), 
                tool_call_id="", 
                status=obj.get("status", ""), 
            ))
        else:
            pass

    return outs 

def sum_usage(usages: Sequence[UsageMetadata]) -> UsageMetadata:
    '''
    计算usages的总和
    '''
    sum_in_tokens = 0
    sum_out_tokens = 0

    for usage in usages:
        sum_in_tokens += usage["input_tokens"]
        sum_out_tokens += usage["output_tokens"]

    return UsageMetadata(
        input_tokens=sum_in_tokens, 
        output_tokens=sum_out_tokens, 
        total_tokens=sum_in_tokens + sum_out_tokens,
    )

def sum_usage_from_messages(messages: Sequence[BaseMessage]) -> UsageMetadata:
    '''
    计算消息中的tokens用量
    '''
    sum_in_tokens = 0
    sum_out_tokens = 0

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.usage_metadata is not None:
            sum_in_tokens = sum_in_tokens + msg.usage_metadata["input_tokens"]
            sum_out_tokens = sum_out_tokens + msg.usage_metadata["output_tokens"]

    return UsageMetadata(
        input_tokens=sum_in_tokens, 
        output_tokens=sum_out_tokens, 
        total_tokens=sum_in_tokens + sum_out_tokens,
    )

def usage_to_output(usage: UsageMetadata) -> Dict:
    return {
        "prompt_tokens": usage["input_tokens"], 
        "completion_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
    }


def estimate_tokens(messages: Sequence[BaseMessage]) -> int:
    '''
    粗略估算消息的tokens数
    '''
    total_tokens = 0

    def _text_tokens(text: str) -> int:
        if not text:
            return 0

        tokens = 0.0
        for c in text:
            # 中文 / CJK
            if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf" or "\uf900" <= c <= "\ufaff":
                tokens += 1.5
            # ASCII 英文 / 数字 / 标点
            elif c.isascii():
                tokens += 0.25
            # Emoji / 其他符号
            elif unicodedata.category(c).startswith("So") or unicodedata.category(c).startswith("Sk"):
                tokens += 1.5
            else:
                # 其他 Unicode 字符（如希腊字母、拉丁扩展等）
                tokens += 1
        return max(1, int(tokens))
        
    for msg in messages:
        total_tokens += 6   # 每条 message 的基础开销
        if isinstance(msg, HumanMessage):
            total_tokens += _text_tokens(msg.content)
        elif isinstance(msg, AIMessage):
            total_tokens += _text_tokens(msg.content)
            for tc in getattr(msg, "tool_calls", []) or []:
                total_tokens += _text_tokens(tc["name"])
                if tc["args"]:
                    args_str = json.dumps(tc["args"], ensure_ascii=False)
                    total_tokens += _text_tokens(args_str)
        elif isinstance(msg, ToolMessage):
            total_tokens += _text_tokens(msg.content)
            if msg.name:
                total_tokens += _text_tokens(msg.name)
        else:
            pass

    # assistant 回复起始 token
    if messages:
        total_tokens += 2

    return total_tokens

