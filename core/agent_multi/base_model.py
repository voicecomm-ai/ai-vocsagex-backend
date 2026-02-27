from typing import Dict, List, TypedDict, Optional, TypeVar
from typing_extensions import Annotated
from abc import ABC, abstractmethod

from pydantic import BaseModel, model_validator, Field
from langchain_core.messages import BaseMessage, UsageMetadata
from langgraph.graph import add_messages

class PlannerOutputItem(BaseModel):
    task_name: str = Field(..., description="子任务名称")
    responsible_agent: str = Field(..., description="负责的子智能体名称")
    input_fields: List[str] = Field(..., description="共享状态，上游子任务输出存放的变量名")
    output_fields: List[str] = Field(..., description="共享状态，存放子智能体执行输出")
    dependencies: List[str] = Field(..., description="依赖的上游子任务")
    prompt_template: str = Field(..., description="子智能体用户提示词")

    @model_validator(mode="after")
    def check(self, ) -> "PlannerOutputItem":
        if not self.task_name:
            raise ValueError("In planner output, `task_name` cannot be a null value.")
        if not self.responsible_agent:
            raise ValueError("In planner output, `responsible_agent` cannot be a null value.")
        if not self.output_fields:
            raise ValueError("In planner output, `output_fields` cannot be a null value.")
        if not self.prompt_template:
            raise ValueError("In planner output, `prompt_template` cannot be a null value.")
        return self

class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(..., description="是否通过反思")
    issues: List[str] = Field(..., description="反思上游结果中存在的问题或不足")
    suggestions: List[str] = Field(..., description="改进措施")
    corrected_text: str = Field(..., description="修正后的文本")

class SubTaskResult(BaseModel):
    task_name: str = Field(..., description="子任务名称")
    agent_name: str = Field(..., description="子智能体名称")
    task_output: str = Field(..., description="子任务输出")

T = TypeVar("T")
def append_list(
    left: Optional[List[T]],
    right: List[T],
) -> List[T]:
    if not right:
        return left or []
    if left is None:
        return list(right)
    return list(left) + list(right)

K = TypeVar("K")
V = TypeVar("V")
def merge_dict(
    left: Optional[Dict[K, V]],
    right: Dict[K, V],
) -> Dict[K, V]:
    if not right:
        return left or {}
    if left is None:
        return dict(right)
    return {**left, **right}

class ManagerOuterState(TypedDict):
    # 用户查询
    query: str
    # 用户输入的聊天历史
    input_history: List[BaseMessage]
    # 本次调用记录的聊天历史
    output_history: List[BaseMessage]
    # 执行计划的字符串
    execution_plan_str: str
    # 执行计划对象列表
    execution_plan: List[PlannerOutputItem]
    # 当前环的重试次数
    retry_times: int
    # 总重试次数
    sum_retry_times: int
    # 子任务输出
    sub_task_outputs: List[SubTaskResult]
    # Final Answer
    answer: str
    # tokens用量列表
    usages: Annotated[List[UsageMetadata], append_list]

class ManagerSubGraphState(TypedDict):
    # 子图的共享状态
    arg_fields: Annotated[Dict[str, str], merge_dict]
    # 子图的任务输出
    task_outputs: Annotated[List[SubTaskResult], append_list]
    # 子图的tokens用量列表
    usages: Annotated[List[UsageMetadata], append_list]

class MultiAgentSubAdditionalKwargs(BaseModel, ABC):
    @abstractmethod
    def to_dict(self, ) -> Dict:
        raise NotImplementedError
    
class ManagerMultiAgentSubAdditionalKwargs(MultiAgentSubAdditionalKwargs):
    task_name: str
    agent_name: str
    agent_content: str
    agent_status: str

    def to_dict(self, ) -> Dict:
        return {
            "task_name" : self.task_name, 
            "agent_name" : self.agent_name, 
            "agent_content" : self.agent_content, 
            "agent_status" : self.agent_status, 
        }