from typing import (
    Optional, List, Any, Sequence, 
    Generator, AsyncGenerator, 
    Callable, Awaitable, Dict, 
    cast, 
)
from concurrent.futures.thread import ThreadPoolExecutor
import traceback
import asyncio

from langchain.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.state import (
    CompiledStateGraph, StateGraph, END, START, 
)
from langgraph.config import get_stream_writer
from jinja2 import Template, Undefined

from core.agent_multi.base_multi_agent import BaseMultiAgent
from core.agent.base_agent import BaseAgent
from core.agent.memory_agent import MemoryAgent
from core.agent.base_model  import AgentResult
from core.agent_multi.base_model import (
    ManagerOuterState, 
    ManagerSubGraphState, 
    PlannerOutputItem, 
    SubTaskResult, 
    ManagerMultiAgentSubAdditionalKwargs, 
)
from core.agent_multi.base_constants import (
    PLANNER_SYSTEM_PROMPT, 
    PLANNER_HUMAN_PROMPT_TEMPLATE, 
    REFLECTION_SYSTEM_PROMPT, 
    REFLECTION_HUMAN_PROMPT_TEMPLATE, 
    AGGREGATOR_SYSTEM_PROMPT, 
    AGGREGATOR_HUMAN_PROMPT_TEMPLATE, 
)
from core.agent_multi.base_utils import (
    generate_reflection, 
    generate_execution_plan, 
    generate_msg_from_execution_plan, 
    generate_msgs_from_subtask_result, 
)
from core.agent.base_utils import (
    sum_usage, 
    get_history_turn, 
    truncate_history_front, 
    simplify_history, 
)
from core.model.model_utils import (
    remove_think_tag
)
from logger import get_logger

logger = get_logger("multi")

class ManagerMultiAgent(BaseMultiAgent):
    def __init__(
        self, 
        name: str, 
        description: str, 
        recursion_limit: int = 10, 
        depth: int = 10, 
        summarize: bool = False, 
        summarize_limit: int = 2048, 
        summarize_model: Optional[BaseChatModel] = None, 
        retry_times: int = 0, 
        reflect_manager: bool = True, 
        reflect_sub_agent: bool = False,  # 暂不支持
        **kwargs
    ):
        self.name = name
        self.description = description
        self._retry_times = retry_times
        self._reflect_manager = reflect_manager
        self._reflect_sub_agent = False # 暂不支持

        self.sub_agents: Dict[str, BaseAgent] = {}

        # 创建内部智能体
        self.backup_agent = MemoryAgent(
            name="_Backup_Agent", 
            description="备用智能体，用于任务无法拆解时的简单问答。", 
            recursion_limit=recursion_limit, 
            depth=depth, 
            summarize=summarize, 
            summarize_limit=summarize_limit, 
            summarize_model=summarize_model, 
        )

        self.planner_agent = MemoryAgent(
            name="_Planner_Agent", 
            description="主管智能体，用于子任务拆分和执行计划生成。", 
            recursion_limit=recursion_limit, 
            depth=depth, 
            summarize=summarize, 
            summarize_limit=summarize_limit, 
            summarize_model=summarize_model, 
        )
        self.planner_agent.set_system_prompt(PLANNER_SYSTEM_PROMPT)

        self.aggregator_agent = BaseAgent(
            name="_Aggregator_Agent", 
            description="结果汇总智能体，用于将子任务输出整合为最终结果。", 
            recursion_limit=recursion_limit, 
        )
        self.aggregator_agent.set_system_prompt(AGGREGATOR_SYSTEM_PROMPT)

        if self._reflect_manager or self._reflect_sub_agent:
            self.reflection_agent = BaseAgent(
                name="_Reflection_Agent", 
                description="反思智能体，用于评审和修正上游生成结果。", 
                recursion_limit=recursion_limit, 
            )
            self.reflection_agent.set_system_prompt(REFLECTION_SYSTEM_PROMPT)
        else:
            self.reflection_agent = None

        self._inner_agents_name = [
            self.backup_agent.name, 
            self.planner_agent.name, 
            self.aggregator_agent.name, 
        ]
        if self.reflection_agent:
            self._inner_agents_name.append(self.reflection_agent.name)

        # 生成内部执行图结构
        self.main_graph = StateGraph(ManagerOuterState)
        self._build_main_graph()
        self.multi_agent: Optional[CompiledStateGraph] = None

        # 所有的图包括子图
        # 流式数据，均通过custom格式上抛
        # 节点执行state，通过values上抛
        # 自定义数据格式为{"custom_data": AgentResult}

    def set_chat_model(
        self, 
        planner_chat_model: BaseChatModel, 
        backup_chat_model: BaseChatModel, 
        aggregator_chat_model: BaseChatModel, 
        reflection_chat_model: Optional[BaseChatModel] = None, 
    ):
        self.backup_agent.set_chat_model(backup_chat_model)
        self.planner_agent.set_chat_model(planner_chat_model)
        self.aggregator_agent.set_chat_model(aggregator_chat_model)
        if self.reflection_agent:
            self.reflection_agent.set_chat_model(reflection_chat_model)

    def set_memory(
        self, 
        retrieve_func: Optional[Callable[[str], List[str]]] = None, 
        retrieve_afunc: Optional[Callable[[str], Awaitable[List[str]]]] = None, 
        record_func: Optional[Callable[[str, List[str]], Any]] = None, 
        record_afunc: Optional[Callable[[str, List[str]], Awaitable[Any]]] = None, 
        thread_pool: Optional[ThreadPoolExecutor] = None, 
    ):
        self.backup_agent.set_memory(
            retrieve_func=retrieve_func, 
            retrieve_afunc=retrieve_afunc, 
            record_func=None,           # planner 会执行记录，故backup不再重复记录
            record_afunc=None,          # planner 会执行记录，故backup不再重复记录
            thread_pool=thread_pool, 
        )
        self.planner_agent.set_memory(
            retrieve_func=retrieve_func, 
            retrieve_afunc=retrieve_afunc, 
            record_func=record_func, 
            record_afunc=record_afunc, 
            thread_pool=thread_pool, 
        )

    def set_sub_agents(self, sub_agents: List[BaseAgent]):
        if not sub_agents:
            raise ValueError("Empty sub agents list.")

        for sub in sub_agents:
            if sub.name in self.sub_agents:
                raise ValueError(f"Repeated agent name [{sub.name}].")
            
            if sub.name in self._inner_agents_name:
                raise ValueError(f"Agent [{sub.name}] used internal name.")

            self.sub_agents[sub.name] = sub

    def compile(self, ):
        self.backup_agent.compile()
        self.planner_agent.compile()
        self.aggregator_agent.compile()
        if self.reflection_agent:
            self.reflection_agent.compile()

        if self.multi_agent is None:
            self.multi_agent = self.main_graph.compile()

    def invoke(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, **kwargs
    ) -> AgentResult:
        raise NotImplementedError

    async def ainvoke(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, **kwargs
    ) -> AgentResult:
        raise NotImplementedError

    def invoke_stream(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, **kwargs
    ) -> Generator[AgentResult, None, None]:
        raise NotImplementedError

    async def ainvoke_stream(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, **kwargs
    ) -> AsyncGenerator[AgentResult, None]:
        
        gen = self.multi_agent.astream(
            input={
                "query": query, 
                "input_history": history or [], 
                "retry_times": 0, 
                "sum_retry_times": 0, 
            }, 
            stream_mode=["custom", "values"]
        )

        latest_state = None
        async for chunk in gen:
            if chunk[0] == "custom":
                data = cast(AgentResult, chunk[1]["custom_data"])
                agent_name = data.additional_kwargs.get("agent_name")
                if agent_name == self.aggregator_agent.name or agent_name == self.backup_agent.name: 
                    # 产出 Final Answer 的智能体
                    if data.assistant_response:
                        # 为了过滤，汇总智能体的末尾chunk，该chunk是表示智能体的结束
                        # 不是多智能体的结束，多智能体的结束chunk在循环外部生成
                        # 移除 Final Answer 的 agent_name 字段
                        data.additional_kwargs.pop("agent_name")
                        yield AgentResult(
                            assistant_response=data.assistant_response, 
                        )
                elif agent_name in self.sub_agents:
                    # 子智能体
                    yield AgentResult(
                        assistant_response="", 
                        additional_kwargs=ManagerMultiAgentSubAdditionalKwargs(
                            task_name=data.additional_kwargs.get("task_name"), 
                            agent_name=data.additional_kwargs.get("agent_name"), 
                            agent_content=data.assistant_response, 
                            agent_status="FINISHED" if data.done else "STREAM", 
                        ).to_dict(), 
                    )
                elif agent_name == "__Inner_Message":
                    # 给前端的额外响应
                    yield AgentResult(
                        assistant_response="", 
                        additional_kwargs={
                            "is_subtask_done": data.additional_kwargs.get("is_subtask_done"), 
                        }
                    )
                else:
                    # Other
                    pass
            elif chunk[0] == "values":
                latest_state = chunk[1]
        
        # 汇总usage
        s_usage = sum_usage(latest_state["usages"])

        # 发回末尾消息
        yield AgentResult(
            assistant_response="", 
            done=True, 
            additional_kwargs={}, 
            usage=s_usage, 
            history=latest_state["output_history"] if latest_state else []
        )

    def _build_main_graph(self, ):
        self.main_graph.add_node("planner", self._a_planner_node)
        self.main_graph.add_node("reflection_planner", self._a_reflect_planner_node)
        self.main_graph.add_node("dynamic", self._a_dynamic_graph_node)
        self.main_graph.add_node("aggregator", self._a_aggregator_node)
        self.main_graph.add_node("backup", self._a_backup_node)

        if self._reflect_manager:
            # 对planner进行反思的图结构
            #             ↑→------------------------------------→ backup →------------------------------→↓
            #             |                                          ↑                                   |
            #             ↑                                          ↑                                   ↓
            # planner → check_planner → reflection_planner → check_before_dynamic → dynamic → aggregator → END
            # ↑           ↓                                          ↓
            # |←---------←↓                                          ↓
            # ↑←----------------------------------------------------←↓
            self.main_graph.add_conditional_edges(
                "planner", self._a_check_planner, 
                {
                    "planner": "planner", 
                    "backup": "backup", 
                    "reflection_planner": "reflection_planner", 
                }
            )
            self.main_graph.add_conditional_edges(
                "reflection_planner", self._a_check_before_dynamic, 
                {
                    "backup": "backup", 
                    "dynamic": "dynamic", 
                    "planner": "planner", 
                }
            )
            self.main_graph.add_edge("backup", END)
            self.main_graph.add_edge("dynamic", "aggregator")
            self.main_graph.add_edge("aggregator", END)

        else:
            # 不反思的图结构
            #                    ↑→------------→ backup →-------------→↓
            #                    |                                     |
            #                    ↑                                     ↓
            # planner → check_before_dynamic → dynamic → aggregator → END
            #    ↑               ↓
            #    ↑←-------------←↓
            self.main_graph.add_conditional_edges(
                "planner", self._a_check_before_dynamic, 
                {
                    "backup": "backup", 
                    "dynamic": "dynamic", 
                    "planner": "planner", 
                }
            )
            self.main_graph.add_edge("backup", END)
            self.main_graph.add_edge("dynamic", "aggregator")
            self.main_graph.add_edge("aggregator", END)

        self.main_graph.set_entry_point("planner")

        return 

    async def _a_planner_node(self, state: ManagerOuterState) -> Dict:
        '''
        Planner 节点
        '''
        logger.debug(f"{self.name} Enter 'planner' node.")

        query = state["query"]
        history = state["input_history"]
        retry_times = state["retry_times"] if "retry_times" in state else 0
        sum_retry_times = state["sum_retry_times"] if "sum_retry_times" in state else 0

        execution_plan_str = ""
        execution_plan = []

        try:
            gen = self.planner_agent.ainvoke_stream(
                query=Template(PLANNER_HUMAN_PROMPT_TEMPLATE, undefined=Undefined).render(
                    **{
                        "query": query, 
                        "agents": {
                            name: sub.description 
                            for name, sub in self.sub_agents.items()
                        }
                    }
                ), 
                memory_query=query, 
                history=history, 
                record=True if sum_retry_times == 0 else False, # planner重试时，不再记录长期记忆
            )

            first_chunk = True
            stream_writer = get_stream_writer()
            async for chunk in gen:
                if first_chunk:
                    logger.debug(f"{self.name} Planner: Generating...")
                    first_chunk = False

                stream_writer({"custom_data": chunk})
                execution_plan_str += chunk.assistant_response
            usage = chunk.usage
        except Exception as e:
            logger.error(f"{self.name} Fail to invoke planner for:\n{traceback.format_exc()}")
            raise RuntimeError(f"Planner: Fail to invoke, please check whether your chat model is available.")

        logger.debug(f"{self.name} Original Execution plan str:\n{execution_plan_str}")
        execution_plan_str = remove_think_tag(execution_plan_str)
        logger.debug(f"{self.name} Execution plan str:\n{execution_plan_str}")

        try:
            execution_plan = generate_execution_plan(
                execution_plan_str, 
                [name for name, _ in self.sub_agents.items()], 
            )

            # 提取切分、摘要后的历史
            output_history_turn = get_history_turn(chunk.history)
            output_history = truncate_history_front(chunk.history, output_history_turn - 1)
            # 插入query
            output_history.append(HumanMessage(content=query))
            
        except Exception as e:
            retry_times = retry_times + 1
            sum_retry_times = sum_retry_times + 1
            logger.debug(f"{self.name} Planner[{retry_times}][{sum_retry_times}] Fail to generate execution plan for:\n{traceback.format_exc()}")
            return {
                "execution_plan_str": "", 
                "execution_plan": [], 
                "retry_times": retry_times, 
                "usages": [usage] if usage else [], 
                "sum_retry_times": sum_retry_times, 
            }
        else:
            return {
                "execution_plan_str": execution_plan_str, 
                "execution_plan": execution_plan, 
                "retry_times": 0, # 成功时，清除当前环的重试次数
                "usages": [usage] if usage else [], 
                "output_history": output_history if output_history else [],  # 每次进入planner，聊天历史都重新生成
            }

    async def _a_reflect_planner_node(self, state: ManagerOuterState) -> Dict:
        '''
        反思 Planner 节点
        '''
        logger.debug(f"{self.name} Enter 'reflection_planner' node.")

        query = state["query"]
        retry_times = state["retry_times"]
        sum_retry_times = state["sum_retry_times"]
        execution_plan_str = state["execution_plan_str"]

        reflection_str = ""
        execution_plan = []

        try:
            gen = self.reflection_agent.ainvoke_stream(
                query=Template(REFLECTION_HUMAN_PROMPT_TEMPLATE, undefined=Undefined).render(
                    **{
                        "query": f"请将“{query}”拆解成多个子任务，并生成合适的执行计划。", 
                        "generated_result": execution_plan_str,
                    }
                )
            )

            first_chunk = True
            stream_writer = get_stream_writer()
            async for chunk in gen:
                if first_chunk:
                    logger.debug(f"{self.name} Reflection: Generating...")
                    first_chunk = False

                stream_writer({"custom_data": chunk})
                reflection_str += chunk.assistant_response
            usage = chunk.usage
        except Exception as e:
            logger.error(f"{self.name} Fail to invoke reflection for:\n{traceback.format_exc()}")
            raise RuntimeError(f"Reflection: Fail to invoke, please check whether your chat model is available.")

        logger.debug(f"{self.name} Reflection result:\n{reflection_str}")

        try:
            reflection_obj = generate_reflection(reflection_str)
            if reflection_obj.is_satisfactory == False:
                # 上游节点结果 未通过反思节点
                # 重新生成执行计划
                execution_plan = generate_execution_plan(
                    reflection_obj.corrected_text, 
                    [name for name, _ in self.sub_agents.items()], 
                )

                return {
                    "execution_plan_str": reflection_obj.corrected_text, 
                    "execution_plan": execution_plan, 
                    "retry_times": 0, # 成功时，清除当前环的重试次数
                    "usages": [usage] if usage else [], 
                }
            else:
                # 上游节点结果 通过反思节点
                return {
                    "retry_times": 0, # 成功时，清除当前环的重试次数
                    "usages": [usage] if usage else [], 
                }
        except Exception as e:
            retry_times = retry_times + 1
            sum_retry_times = sum_retry_times + 1
            logger.debug(f"{self.name} Reflection[{retry_times}][{sum_retry_times}] Fail to generate execution plan for:\n{traceback.format_exc()}")
            return {
                "execution_plan_str": "", 
                "execution_plan": [], 
                "retry_times": retry_times, 
                "usages": [usage] if usage else [], 
                "sum_retry_times": sum_retry_times, 
            }

    async def _a_aggregator_node(self, state: ManagerOuterState) -> Dict:
        '''
        汇总 节点
        '''
        logger.debug(f"{self.name} Enter 'aggregator' node.")

        query = state["query"]
        sub_task_outputs = state["sub_task_outputs"]
        output_history = state["output_history"]

        answer = ""

        try:
            gen = self.aggregator_agent.ainvoke_stream(
                query=Template(AGGREGATOR_HUMAN_PROMPT_TEMPLATE, undefined=Undefined).render(
                    **{
                        "query": query, 
                        "task_outputs": {item.task_name: item.task_output for item in sub_task_outputs}
                    }
                )
            )

            first_chunk = True
            stream_writer = get_stream_writer()
            async for chunk in gen:
                if first_chunk:
                    logger.debug(f"{self.name} Aggregator: Generating...")
                    first_chunk = False

                stream_writer({"custom_data": chunk})
                answer += chunk.assistant_response
            usage = chunk.usage
        except Exception as e:
            logger.error(f"{self.name} Fail to invoke aggregator for:\n{traceback.format_exc()}")
            raise RuntimeError(f"Aggregator: Fail to invoke, please check whether your chat model is available.")

        # 输出的聊天历史中补上Final Answer
        output_history = output_history + [AIMessage(content=answer)]
        output_history = simplify_history(output_history)

        return {
            "answer": answer, 
            "usages": [usage] if usage else [], 
            "output_history": output_history, 
        }

    async def _a_backup_node(self, state: ManagerOuterState) -> Dict:
        '''
        备用 节点
        '''
        logger.debug(f"{self.name} Enter 'backup' node.")

        query = state["query"]
        input_history = state["input_history"]

        answer = ""

        try:
            gen = self.backup_agent.ainvoke_stream(
                query=query, 
                history=input_history, 
            )

            first_chunk = True
            stream_writer = get_stream_writer()
            async for chunk in gen:
                if first_chunk:
                    logger.debug(f"{self.name} Backup: Generating...")
                    first_chunk = False

                stream_writer({"custom_data": chunk})
                answer += chunk.assistant_response
            usage = chunk.usage
        except Exception as e:
            logger.error(f"{self.name} Fail to invoke backup for:\n{traceback.format_exc()}")
            raise RuntimeError(f"Backup: Fail to invoke, please check whether your chat model is available.")

        # 执行backup节点时，真实有效的过程只有一个backup节点，所以直接使用backup智能体返回的聊天历史
        output_history = simplify_history(chunk.history)

        return {
            "answer": answer, 
            "usages": [usage] if usage else [], 
            "output_history": output_history, 
        }

    async def _a_dynamic_graph_node(self, state: ManagerOuterState) -> Dict:
        '''
        动态图 节点
        '''
        logger.debug(f"{self.name} Enter 'dynamic' node.")

        query = state["query"]
        execution_plan = state["execution_plan"]
        output_history = state["output_history"]

        # 根据 execution planner 构建动态图
        try:
            compiled_graph = self._build_subgraph_from_planner(state["execution_plan"])
        except Exception as e:
            logger.error(f"{self.name} Fail to build subgraph for:\n{traceback.format_exc()}")
            raise
        
        try:
            gen = compiled_graph.astream(
                input={
                    "arg_fields": {
                        "query": query, 
                    }, 
                }, 
                stream_mode=["custom", "values"], 
            )

            sub_latest_state = None
            stream_writer = get_stream_writer()
            async for chunk in gen:
                if chunk[0] == "custom":
                    stream_writer({"custom_data": chunk[1]["custom_data"]})
                elif chunk[0] == "values":
                    # 取最后一次子图的输出的state
                    sub_latest_state = chunk[1]
    
            # 前端需要，当所有子任务结束后发送子任务结束字段，使用额外智能体名称 __Inner_Message
            stream_writer({"custom_data": AgentResult(
                assistant_response="", 
                done=True, 
                additional_kwargs={
                    "agent_name": "__Inner_Message", 
                    "is_subtask_done": True, 
                }
            )})

            # 构建剩余的聊天历史
            # 执行计划：模拟为 AIMessage
            msgs_0 = [generate_msg_from_execution_plan(execution_plan)]
            #  子任务输出：模拟为 ToolMessage
            msgs_1 = generate_msgs_from_subtask_result(sub_latest_state["task_outputs"])
        except Exception as e:
            logger.error(f"{self.name} Fail to run subgraph for {type(e).__name__}:{e}")
            raise

        return {
            "sub_task_outputs": sub_latest_state["task_outputs"], 
            "usages": sub_latest_state["usages"], 
            "output_history": output_history + msgs_0 + msgs_1, 
        }

    async def _a_check_planner(self, state: ManagerOuterState) -> str:
        '''
        Planner 校验路由
        '''
        logger.debug(f"{self.name} Enter 'check_planner' route.")

        retry_times = state["retry_times"]
        execution_plan = state["execution_plan"]
        sum_retry_times = state["sum_retry_times"]

        if retry_times > self._retry_times:
            # planner 超过重试次数上限
            return "backup"
        elif retry_times == 0:
            # planner 通过
            if len(execution_plan) == 0:
                # 未拆分出有效的子任务
                return "backup"
            else:
                # 拆分出有效的子任务
                return "reflection_planner"
        else:
            if sum_retry_times > self._retry_times:
                # 总重试次数 超过重试次数上限
                return "backup"
            else:
                # planner 重试中
                return "planner"
    
    async def _a_check_before_dynamic(self, state: ManagerOuterState) -> str:
        '''
        动态图前 校验路由
        '''
        logger.debug(f"{self.name} Enter 'check_before_dynamic' route.")

        retry_times = state["retry_times"]
        execution_plan = state["execution_plan"]
        sum_retry_times = state["sum_retry_times"]

        if retry_times > self._retry_times:
            # planner 超过重试次数
            return "backup"
        elif retry_times == 0:
            if len(execution_plan) == 0:
                # 未拆分出有效的子任务
                return "backup"
            else:
                # 拆分出有效的子任务
                return "dynamic"
        else:
            if sum_retry_times > self._retry_times:
                # 总重试次数 超过重试次数上限
                return "backup"
            else:
                # planner 重试中
                return "planner"
        
    def _build_subgraph_from_planner(
        self, 
        excution_plan: List[PlannerOutputItem], 
    ) -> CompiledStateGraph:
        graph = StateGraph(ManagerSubGraphState)

        # 添加节点
        for plan_item in excution_plan:
            logger.debug(f"{self.name} subgraph add node [{plan_item.task_name}].")
            graph.add_node(
                plan_item.task_name, 
                self._generate_subgraph_node_afunc(plan_item), 
            )

        # 添加边
        for plan_item in excution_plan:
            if not plan_item.dependencies:
                logger.debug(f"{self.name} subgraph add edge [{START}] -> [{plan_item.task_name}].")
                graph.add_edge(START, plan_item.task_name)
            else:
                for dep in plan_item.dependencies:
                    logger.debug(f"{self.name} subgraph add edge [{dep}] -> [{plan_item.task_name}].")
                    graph.add_edge(dep, plan_item.task_name)

        # 添加结束任务
        all_tasks = {t.task_name for t in excution_plan}
        depended_tasks = {d for t in excution_plan for d in t.dependencies}
        end_tasks = all_tasks - depended_tasks

        for task_name in end_tasks:
            logger.debug(f"{self.name} subgraph add edge [{task_name}] -> [{END}].")
            graph.add_edge(task_name, END)

        return graph.compile()

    def _generate_subgraph_node_afunc(
        self, 
        plan_item: PlannerOutputItem, 
    ) -> Callable[[ManagerSubGraphState], Awaitable[Dict]]:
        # 找到子智能体
        sub_agent = self.sub_agents[plan_item.responsible_agent]

        async def node_afunc(state: ManagerSubGraphState) -> Dict:
            logger.debug(f"{self.name} Enter subgraph '{plan_item.responsible_agent}-{plan_item.task_name}' node.")

            arg_fields = state["arg_fields"]

            task_output = ""

            input_fields = {
                arg: arg_fields[arg] for arg in plan_item.input_fields
            }

            try:
                gen = sub_agent.ainvoke_stream(
                    query=Template(plan_item.prompt_template, undefined=Undefined).render(
                        **input_fields
                    )
                )

                first_chunk = True
                stream_writer = get_stream_writer()
                async for chunk in gen:
                    if first_chunk:
                        logger.debug(f"{self.name} {sub_agent.name}: Generating...")
                        first_chunk = False

                    # 动态图 子智能体流式响应 additional_kwargs 中追加 task_name
                    chunk.additional_kwargs["task_name"] = plan_item.task_name
                    stream_writer({"custom_data": chunk})
                    task_output += chunk.assistant_response
                
                # usage 存在最后一个chunk中
                usage = chunk.usage
            except Exception as e:
                logger.error(f"{self.name} subgraph '{plan_item.responsible_agent}-{plan_item.task_name}' Error:\n{traceback.format_exc()}")
                raise RuntimeError(f"{plan_item.responsible_agent}-{plan_item.task_name}: Fail to invoke.")
            except asyncio.CancelledError:
                raise

            logger.debug(f"{self.name} subgraph '{plan_item.responsible_agent}-{plan_item.task_name}' Output:\n{task_output}")

            return {
                "arg_fields": {
                    plan_item.output_fields[0]: task_output
                }, 
                "task_outputs" : [
                    SubTaskResult(
                        task_name=plan_item.task_name, 
                        agent_name=plan_item.responsible_agent, 
                        task_output=task_output, 
                    )
                ],
                "usages": [usage] if usage else [], 
            }

        return node_afunc
