from typing import Any, List, Awaitable, Callable, Dict
import re
import json

from langgraph.graph.state import CompiledStateGraph, StateGraph, START, END
from jinja2 import Template, StrictUndefined
from langchain.messages import AIMessage, ToolCall, ToolMessage

from core.agent_multi.base_model import PlannerOutputItem, ReflectionOutput, SubTaskResult
from core.agent.base_agent import BaseAgent

def load_llm_json(text: str) -> Any:
    # 直接加载
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = text.strip()

    # 移除markdown的json标记
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    # 提取第一个合法json对象或数组
    match = re.search(
        r"(\{.*\}|\[.*\])",
        cleaned,
        flags=re.DOTALL
    )
    if not match:
        raise RuntimeError(f"No json object or array in the output of planner:\n{text}")

    json_text = match.group(1)

    # 重新加载
    return json.loads(json_text)

def generate_execution_plan(text: str, available_agents: List[str]) -> List[PlannerOutputItem]:
    # 加载 JSON
    obj = load_llm_json(text)

    # JSON 字段检查
    if not isinstance(obj, list):
        raise RuntimeError(f"The output of planner is not a json array:\n{text}.")

    outputs = []
    sub_tasks = []
    shared_state=["query", ]

    for arr_item in obj:
        # 1. 检查字段名称、类型是否合规，齐全
        item = PlannerOutputItem(**arr_item)

        # 2. 检查智能体名称是否合法
        if item.responsible_agent not in available_agents:
            raise RuntimeError(f"Agent '{item.responsible_agent}' is not available.")

        # 3. 检查入参是否存在
        if not set(item.input_fields).issubset(shared_state):
            raise RuntimeError(f"Undefined `input_fields` when required:\n{item.input_fields}")

        # 4. 检查出参是否重复
        if set(item.output_fields) & set(shared_state):
            raise RuntimeError(f"Duplicate `output_fields` detected:\n{item.output_fields}")
        shared_state.extend(item.output_fields)

        # 5. 检查子任务依赖的子任务是否存在，此处可检查是否有环
        if not set(item.dependencies).issubset(sub_tasks):
            raise RuntimeError(f"Depends on non-existent `dependencies`.:\n{item.dependencies}")
        sub_tasks.append(item.task_name)
    
        # 6. 检查用户提示词模板中的变量是否在入参中存在
        temp_dict = {k: "" for k in shared_state}
        try:
            Template(item.prompt_template, undefined=StrictUndefined).render(**temp_dict)
        except Exception as e:
            raise RuntimeError(f"Found undefined variables in `prompt_template`:\n\t{type(e).__name__}{e}\n{item.prompt_template}")

        outputs.append(item)

    return outputs

def generate_reflection(text: str) -> ReflectionOutput:
    obj = load_llm_json(text)
    return ReflectionOutput(**obj)

def generate_msg_from_execution_plan(plan: List[PlannerOutputItem]) -> AIMessage:
    '''
    将 执行计划 伪造为含 ToolCall 的 AIMessage
    '''
    tool_calls = []
    
    for item in plan:
        tool_calls.append(
            ToolCall(
                name=item.responsible_agent, 
                args={
                    "query": item.prompt_template, 
                }, 
                id=None, 
            )
        )

    return AIMessage(
        content="", 
        tool_calls=tool_calls, 
    )

def generate_msgs_from_subtask_result(results: List[SubTaskResult]) -> List[ToolMessage]:
    out = []
    for result in results:
        out.append(
            ToolMessage(
                content=result.task_output, 
                name=result.agent_name, 
                tool_call_id="", 
                status="success", 
            )
        )

    return out