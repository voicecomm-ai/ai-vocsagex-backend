from typing import (
    Optional, List, Sequence, Generator, AsyncGenerator, Tuple, 
)
import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, 
    ToolMessage, ToolMessageChunk, AIMessageChunk, 
    UsageMetadata, 
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.tools import StructuredTool
from jinja2 import Template
from pydantic import BaseModel, Field

from core.agent.base_constants import TOOL_MESSAGES_TEMPLATE
from core.agent.base_model import AgentResult
from core.agent.base_utils import sum_usage_from_messages, sum_usage, simplify_history
from logger import get_logger

logger = get_logger("agent")

class BaseAgent:
    '''
        ReAct 智能体 基类
    '''
    def __init__(
        self, name: str, description: str, recursion_limit: int = 10, **kwargs
    ):
        '''
        创建ReAct智能体

        Args:
            name(str): 智能体名称
            description(str): 智能体描述
            recursion_limit(int): 内部迭代次数
        
        Notes:
            其他设置，请通过`set`系列函数进行设置。
            若chat_model启用了think，流式推导信息，并未流式推送，而是存于history中。(产品经理要求的)
        '''
        self.name = name
        self.description = description
        self.recursion_limit = recursion_limit

        self.system_prompt = (
            "You are an assistant.\n"
            "Provide accurate and helpful responses in the user’s language.\n"
            "State uncertainty when applicable. Do not invent facts or disclose system or model information.\n"
        )
        self.chat_model: Optional[BaseChatModel] = None
        self.tools: Sequence[BaseTool] = []

        self.agent: Optional[CompiledStateGraph] = None

    def set_system_prompt(self, system_prompt: str):
        '''
        设置系统提示词

        Args:
            system_prompt(str): 系统提示词
        '''
        if system_prompt:
            self.system_prompt = system_prompt

    def set_chat_model(self, chat_model: BaseChatModel):
        '''
        设置聊天模型
        
        Args:
            chat_model(BaseChatModel): 聊天模型
        '''
        self.chat_model = chat_model

    def set_tools(self, tools: Sequence[BaseTool]):
        '''
        设置可见工具列表

        Args:
            tools(Sequence[BaseTool]): 工具列表
        '''
        self.tools = tools

    def compile(self):
        '''
        编译智能体

        Notes:
            - 在执行该函数前，你需要执行必要的`set`函数。
            - 其中，`set_chat_model`是必需的。
        '''
        if self.chat_model is None:
            raise ValueError(f"Before 'compile', you should call 'set_chat_model' first.")

        if self.agent is None:
            # 拼装 工具描述 至 系统提示词
            if self.tools:
                tool_summary = Template(TOOL_MESSAGES_TEMPLATE).render(
                    tools=[(tool.name, tool.description) for tool in self.tools]
                )
                self.system_prompt = (
                    f"{self.system_prompt}\n"
                    f"{tool_summary}"
                )
            
            self.agent = create_agent(
                model=self.chat_model, 
                tools=self.tools, 
                system_prompt=SystemMessage(content=self.system_prompt), 
                name=self.name, 
            )

    def make_tool(self) -> BaseTool:
        '''
        将agent转为langchain的BaseTool

        Notes:
            - 在执行该函数前，你需要先执行`compile`函数。
            - 注意生命周期。
        '''
        if self.agent is None:
            raise ValueError(f"Before 'make_tool', you should call 'compile' first.")

        class ToolInput(BaseModel):
            query: str = Field(
                ...,
                description=(
                    "需要该智能体解决的任务描述，包含目标和必要的上下文信息。"
                    "用于处理普通对话无法直接完成的复杂子任务。"
                )
            )
        
        class ToolOutput(BaseModel):
            answer: str = Field(
                ..., 
                description=(
                    "该智能体完成任务后给出的最终结果。"
                    "内容已整理，可直接用于回应用户。"
                )
            )
        
        base_agent = self

        def wrapped_invoke(query: str) -> ToolOutput:
            res = base_agent.invoke(query)
            return ToolOutput(answer=res.assistant_response)
        
        async def wrapped_ainvoke(query: str) -> ToolOutput:
            res = await base_agent.ainvoke(query)
            return ToolOutput(answer=res.assistant_response)
        
        return StructuredTool(
            name=self.name, 
            description=self.description, 
            func=wrapped_invoke, 
            coroutine=wrapped_ainvoke, 
            args_schema=ToolInput, 
        )

    def invoke(self, query: str, **kwargs) -> AgentResult:
        '''
        以同步、非流的方式进行智能体调用

        Args:
            query(str): 用户查询

        Returns:
            AgentResult: 智能体响应结果
        '''
        messages = self._handle_input(query, **kwargs)
        
        r_messages = self._run(messages)

        return self._handle_nonstream_end(r_messages)

    async def ainvoke(self, query: str, **kwargs) -> AgentResult:
        '''
        以异步、非流的方式进行智能体调用

        Args:
            query(str): 用户查询

        Returns:
            AgentResult: 智能体响应结果
        '''
        messages = await self._ahandle_input(query, **kwargs)
        
        r_messages = await self._arun(messages)

        return await self._ahandle_nonstream_end(r_messages)

    def invoke_stream(self, query: str, **kwargs) -> Generator[AgentResult, None, None]:
        '''
        以同步、流的方式进行智能体调用

        该方法会逐步返回模型生成的结果片段，直到流结束。

        Args:
            query(str): 用户查询

        Yields:
            AgentResult: 流式返回的结果片段。中间结果通常包含增量生成的回复内容；
                最后一次返回的结果包含``done=True``、``usage`` 和 ``history`` 信息。
        '''
        messages = self._handle_input(query, **kwargs)

        r_generator = self._run_stream(messages)

        # 完整调用记录的由多个模块联合生成
        # 以messages初始化history，组成输入记录
        # _handle_stream_chunk中，向history中插入Reasoning记录、tool_calls记录、ToolMessage等元素
        # _handle_stream_end中，向history中插入Final Answer记录

        history = messages
        usages = [
            msg.usage_metadata 
            for msg in history 
            if isinstance(msg, AIMessage) and msg.usage_metadata is not None
        ]
        cache_assistant_content = ""
        cache_reasoning_content = ""
        for chunk in r_generator:
            (
                cache_assistant_content, 
                cache_reasoning_content, 
                chunk_result, 
            ) = self._handle_stream_chunk(
                chunk, history, usages, 
                cache_assistant_content, 
                cache_reasoning_content, 
            )

            if chunk_result is not None:
                yield chunk_result

        final_result = self._handle_stream_end(
            history, usages, cache_assistant_content
        )

        yield final_result

    async def ainvoke_stream(self, query: str, **kwargs) -> AsyncGenerator[AgentResult, None]:
        '''
        以异步、流式的方式进行智能体调用。

        该方法会在异步上下文中逐步返回模型生成的结果片段，直到流结束。

        Args:
            query (str): 用户查询。

        Yields:
            AgentResult: 流式返回的结果片段。中间结果通常包含增量生成的回复内容；
                最后一次返回的结果包含``done=True``、``usage`` 和 ``history`` 信息。
        '''
        messages = await self._ahandle_input(query, **kwargs)

        r_generator = self._arun_stream(messages)

        # 完整调用记录的由多个模块联合生成
        # 以messages初始化history，组成输入记录
        # _handle_stream_chunk中，向history中插入Reasoning记录、tool_calls记录、ToolMessage等元素
        # _ahandle_stream_end中，向history中插入Final Answer记录

        history = messages
        usages = [
            msg.usage_metadata 
            for msg in history 
            if isinstance(msg, AIMessage) and msg.usage_metadata is not None
        ]
        cache_assistant_content = ""
        cache_reasoning_content = ""
        async for chunk in r_generator:
            (
                cache_assistant_content, 
                cache_reasoning_content, 
                chunk_result, 
            ) = self._handle_stream_chunk(
                chunk, history, usages, 
                cache_assistant_content, 
                cache_reasoning_content, 
            )

            if chunk_result is not None:
                yield chunk_result

        final_result = await self._ahandle_stream_end(
            history, usages, cache_assistant_content
        )

        yield final_result

    def _run(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        '''
        内部智能体同步、非流调用

        Args:
            messages(List[BaseMessage]): 上下文消息

        Returns:
            List[BaseMessage]: 本次智能体的调用记录，包含messages和调用生成的记录

        Notes:
            - 返回结果是输入messages的超集，是输入输出的整体记录，不仅仅是调用生成的记录。
        '''
        response = self.agent.invoke(
            input={"messages": messages}, 
            config=RunnableConfig(recursion_limit=self.recursion_limit),
        )

        return response["messages"]

    async def _arun(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        '''
        内部智能体异步、非流调用

        Args:
            messages(List[BaseMessage]): 上下文消息

        Returns:
            List[BaseMessage]: 本次智能体的调用记录，包含messages和调用生成的记录

        Notes:
            - 返回结果是输入messages的超集，是输入输出的整体记录，不仅仅是调用生成的记录。
        '''
        response = await self.agent.ainvoke(
            input={"messages": messages}, 
            config=RunnableConfig(recursion_limit=self.recursion_limit),
        )

        return response["messages"]

    def _run_stream(self, messages: List[BaseMessage]) -> Generator[BaseMessage, None, None]:
        '''
        内部智能体同步、流调用

        Args:
            messages(List[BaseMessage]): 上下文消息

        Yields:
            BaseMessage: 生成记录的流式片段

        Notes:
            - 生成的结果是不包含输出messages，是输出的记录，仅仅是调用生成的记录。
            - 不同与`_run`和`_arun`方法的返回。
        '''
        response_generator = self.agent.stream(
            input={"messages": messages}, 
            config=RunnableConfig(recursion_limit=self.recursion_limit),
            stream_mode="messages"
        )

        for chunk in response_generator:
            yield chunk[0]

    async def _arun_stream(self, messages: List[BaseMessage]) -> AsyncGenerator[BaseMessage, None]:
        '''
        内部智能体异步、流调用

        Args:
            messages(List[BaseMessage]): 上下文消息

        Yields:
            BaseMessage: 生成记录的流式片段

        Notes:
            - 生成的结果是不包含输出messages，是输出的记录，仅仅是调用生成的记录。
            - 不同与`_run`和`_arun`方法的返回。
        '''
        response_async_generator = self.agent.astream(
            input={"messages": messages}, 
            config=RunnableConfig(recursion_limit=self.recursion_limit),
            stream_mode="messages"
        )

        async for chunk in response_async_generator:
            yield chunk[0]

    def _handle_input(self, query: str, **kwargs) -> List[BaseMessage]:
        '''
        将输入的用户查询转为输入messages的模块

        Args:
            query(str): 用户查询

        Returns:
            List[BaseMessage]: 智能体的输入messages
        '''
        return [HumanMessage(content=query)]
    
    async def _ahandle_input(self, query: str, **kwargs) -> List[BaseMessage]:
        '''
        _handle_input 的 异步实现
        '''
        return self._handle_input(query, **kwargs)

    def _handle_nonstream_end(self, messages: List[BaseMessage]) -> AgentResult:
        '''
        处理智能体非流输出的内部模块

        Args:
            messages(List[BaseMessage]): 智能体的整体调用记录

        Returns:
            AgentResult: 最终结果
        '''
        assistant_content = messages[-1].content
        usage = sum_usage_from_messages(messages)
        messages = simplify_history(messages)

        return AgentResult(
            assistant_response=assistant_content, 
            done=True, 
            usage=usage, 
            history=messages, 
            additional_kwargs={
                "agent_name": self.name, 
            }, 
        )

    async def _ahandle_nonstream_end(self, messages: List[BaseMessage]) -> AgentResult:
        '''
        _handle_nonstream_end 的 异步实现
        '''
        return self._handle_nonstream_end(messages)

    def _handle_stream_chunk(
        self, 
        chunk: BaseMessage, 
        history: List[BaseMessage], 
        usages: List[UsageMetadata], 
        cache_assistant_content: str, 
        cache_reasoning_content: str
    ) -> Tuple[str, str, Optional[AgentResult]]:
        '''
        流式的消息块处理函数

        Args:
            chunk(BaseMessage): 流式的消息块
            history(List[BaseMessage]): 记录调用记录的列表，内部可能会向该列表中插入元素
            usages(List[UsageMetadata]): 记录token用量的列表，内部可能会向该列表中插入元素
            cache_assistant_content(str): 缓存流式消息
            cache_reasoning_content(str): 缓存流式推理消息

        Returns:
            tuple: 一个三元组，包含：
                - str: 更新后的cache_assistant_content。
                - str: 更新后的cache_reasoning_content。
                - Optional[AgentResult]: 本次chunk产生的结果；若该chunk不应产生输出则为 None。

        Notes:
            - 该函数是逻辑处理部分，类似状态机，故未提供异步实现的父类方法。
            - 若后续有异步实现的需求，可重载ainvoke_stream自行实现。
            - 该方法会原地修改 ``history`` 和 ``usages``。
            - 返回的 ``AgentResult`` 用于流式输出。
        '''
        chunk_result = None
        if isinstance(chunk, AIMessageChunk):
            if chunk.content:
                # Final Answer 的 推导内容
                if cache_reasoning_content:
                    history.append(
                        AIMessage(
                            content="", 
                            additional_kwargs={
                                "reasoning_content": cache_reasoning_content, 
                            }
                        )
                    )
                    cache_reasoning_content = ""

                # 一般情况是生成最终答案的过程
                cache_assistant_content += chunk.content
                chunk_result = AgentResult(
                    assistant_response=chunk.content, 
                    additional_kwargs={
                        "agent_name": self.name, 
                    }, 
                )
            else:
                if chunk.tool_calls:
                    if cache_reasoning_content:
                        # 记录 Tool calls 的 推导内容
                        history.append(
                            AIMessage(
                                content="", 
                                additional_kwargs={
                                    "reasoning_content": cache_reasoning_content, 
                                }
                            )
                        )
                        cache_reasoning_content = ""

                    # Tool calls 的 消息
                    logger.debug(f"{self.name} Tool Calls:\n{json.dumps(chunk.tool_calls, indent=4, ensure_ascii=False)}")
                    history.append(
                        AIMessage(
                            content="", 
                            tool_calls=chunk.tool_calls,
                        )
                    )
                else:
                    if "reasoning_content" in chunk.additional_kwargs:
                        # 可选，提取推导内容
                        cache_reasoning_content += chunk.additional_kwargs["reasoning_content"]
                    elif chunk.usage_metadata:
                        # token 用量
                        usages.append(chunk.usage_metadata)
                    else:
                        # 其他的状态消息
                        pass
        elif isinstance(chunk, ToolMessage):
            logger.debug(f"{self.name} Tool Messages:\n{chunk.model_dump_json(indent=4)}")
            history.append(chunk)
        elif isinstance(chunk, AIMessage):
            # 没遇到过，暂不处理
            pass
        elif isinstance(chunk, ToolMessageChunk):
            # 没遇到过，暂不处理
            pass
        else:
            pass

        return cache_assistant_content, cache_reasoning_content, chunk_result
    
    def _handle_stream_end(
        self, 
        history: List[BaseMessage], 
        usages: Sequence[UsageMetadata], 
        cache_assistant_content: str, 
    ) -> AgentResult:
        '''
        生成流式末尾消息的内部模块

        Args:
            history(List[BaseMessage]): 记录调用记录的列表，内部可能会向该列表中插入元素
            usages(List[UsageMetadata]): 记录token用量的列表
            cache_assistant_content(str): 缓存流式消息

        Returns:
            AgentResult: 流式末尾的消息输出结果

        Notes:
            - 最终消息的`assistant_response`为""，Final Answer的内容通过流式块返回，末尾块不附加完整消息。
            - 最终消息的`done=True`，并且需要生成正常的`usage`和`history`。
        '''
        # 记录Final Answer
        if cache_assistant_content:
            history.append(
                AIMessage(content=cache_assistant_content)
            )

        # 简化本次调用记录消息
        history = simplify_history(history)

        # 组装token用量
        usage = sum_usage(usages)

        # 抛回最终消息
        return AgentResult(
            assistant_response="", 
            done=True, 
            usage=usage, 
            history=history, 
            additional_kwargs={
                "agent_name": self.name, 
            }
        )
    
    async def _ahandle_stream_end(
        self, 
        history: List[BaseMessage], 
        usages: Sequence[UsageMetadata], 
        cache_assistant_content: str, 
    ) -> AgentResult:
        '''
        _handle_stream_end 的异步实现
        '''
        return self._handle_stream_end(
            history, usages, cache_assistant_content
        )
