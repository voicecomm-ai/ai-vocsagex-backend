from typing import (
    Optional, Any, List, 
    Sequence, Callable, Awaitable, 
    AsyncGenerator, Generator, 
)
from concurrent.futures import ThreadPoolExecutor
import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import (
    SystemMessage, 
)
from jinja2 import Template

from core.agent.base_model import AgentResult
from core.agent.history_agent import HistoryAgent
from core.agent.base_constants import MEMORY_MESSAGE_TEMPLATE

class MemoryAgent(HistoryAgent):
    '''
        含有长期记忆、聊天历史的 ReAct 智能体
    '''
    def __init__(
        self, name: str, description: str, recursion_limit: int = 10, 
        depth: int = 10, summarize: bool = False, summarize_limit: int = 2048, 
        summarize_model: Optional[BaseChatModel] = None, 
        **kwargs
    ):
        '''
        创建含有长期记忆、聊天历史的ReAct智能体

        Args:
            name(str): 智能体名称
            description(str): 智能体描述
            recursion_limit(int): 内部迭代次数
            depth(int): 聊天历史最大轮次，为0时将不生效，生效值需大于2，小于2时将置为2
            summarize(bool): 是否对聊天历史生成摘要
            summarize_limit(int): 聊天历史token限制
            summarize_model(Optional[BaseChatModel]): 生成摘要的聊天模型
        
        Notes:
            - 长期记忆的检索、记录所产生的token均未被记录。
            - 长期记忆的配置请通过`set_memory`进行配置。
            - 若不进行`set_memory`配置，MemoryAgent将退化为HistoryAgent。
            - 长期记忆的检索、记录均只依赖用户查询`query`，若希望增加对`answer`的依赖，请继承该类并重载相关函数。
            - 长期记忆的检索、记录均发生在智能体调用前，长期记忆的记录不会干扰本次智能体调用。
        '''
        
        self.memory_enable = False
        self.memory_retrieve_func: Optional[Callable[[str], List[str]]] = None
        self.memory_retrieve_afunc: Optional[Callable[[str], Awaitable[List[str]]]] = None
        self.memory_record_func: Optional[Callable[[str, List[str]], Any]] = None
        self.memory_record_afunc: Optional[Callable[[str, List[str]], Awaitable[Any]]] = None
        self.memory_thread_pool: Optional[ThreadPoolExecutor] = None

        super().__init__(
            name, description, recursion_limit, 
            depth, summarize, summarize_limit, summarize_model, 
            **kwargs
        )

    def set_memory(
        self, 
        retrieve_func: Optional[Callable[[str], List[str]]] = None, 
        retrieve_afunc: Optional[Callable[[str], Awaitable[List[str]]]] = None, 
        record_func: Optional[Callable[[str, List[str]], Any]] = None, 
        record_afunc: Optional[Callable[[str, List[str]], Awaitable[Any]]] = None, 
        thread_pool: Optional[ThreadPoolExecutor] = None, 
    ):
        '''
        设置长期记忆的检索、记录方法

        Args:
            retrieve_func(Optional[Callable[[str], List[str]]]): 同步检索方法
            retrieve_afunc(Optional[Callable[[str], Awaitable[List[str]]]]): 异步检索方法
            record_func(Optional[Callable[[str, List[str]], Any]]): 同步记录方法
            record_afunc(Optional[Callable[[str, List[str]], Awaitable[Any]]]): 异步记录方法
            thread_pool(Optional[ThreadPoolExecutor]): 线程池，用以承载同步记录方法的后台运行

        Notes:
            - 使用了一个独立的线程池用于长期记忆的记录，无论是同步还是异步。
            - 这四种方法均可以分别设置为`None`，为`None`时，将不执行对应的操作。
            - record系列函数，你需要自行处理异常，record出现异常并不会影响智能体的调用。
            - 设置了`record_func`，可以设置`thread_pool`，若不设置将使用asyncio的默认线程池。
        '''

        self.memory_enable = any(
            f is not None for f in
            (retrieve_func, retrieve_afunc, record_func, record_afunc)
        )
        self.memory_retrieve_func = retrieve_func
        self.memory_retrieve_afunc = retrieve_afunc
        self.memory_record_func = record_func
        self.memory_record_afunc = record_afunc
        self.memory_thread_pool = thread_pool

    def invoke(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, memory_query: Optional[str] = None, record: bool = True, **kwargs
    ) -> AgentResult:
        '''
        以同步、非流的方式进行智能体调用

        Args:
            query(str): 用户查询
            history(Optional[Sequence[BaseMessage]]): 聊天历史
            memory_query(str): 记忆检索的查询
            record(bool): 主观控制是否记录记忆

        Returns:
            AgentResult: 智能体响应结果
        '''
        return super().invoke(query, history=history, memory_query=memory_query, record=record, **kwargs)

    async def ainvoke(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, memory_query: Optional[str] = None, record: bool = True, **kwargs
    ) -> AgentResult:
        '''
        以异步、非流的方式进行智能体调用

        Args:
            query(str): 用户查询
            history(Optional[Sequence[BaseMessage]]): 聊天历史
            memory_query(str): 记忆检索的查询
            record(bool): 主观控制是否记录记忆

        Returns:
            AgentResult: 智能体响应结果
        '''
        return await super().ainvoke(query, history=history, memory_query=memory_query, record=record, **kwargs)

    def invoke_stream(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, memory_query: Optional[str] = None, record: bool = True, **kwargs
    ) -> Generator[AgentResult, None, None]:
        '''
        以同步、流的方式进行智能体调用

        该方法会逐步返回模型生成的结果片段，直到流结束。

        Args:
            query(str): 用户查询
            history(Optional[Sequence[BaseMessage]]): 聊天历史
            memory_query(str): 记忆检索的查询
            record(bool): 主观控制是否记录记忆

        Yields:
            AgentResult: 流式返回的结果片段。中间结果通常包含增量生成的回复内容；
                最后一次返回的结果包含``done=True``、``usage`` 和 ``history`` 信息。
        '''
        yield from super().invoke_stream(query, history=history, memory_query=memory_query, record=record, **kwargs)

    async def ainvoke_stream(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, memory_query: Optional[str] = None, record: bool = True, **kwargs
    ) -> AsyncGenerator[AgentResult, None]:
        '''
        以异步、流式的方式进行智能体调用。

        该方法会在异步上下文中逐步返回模型生成的结果片段，直到流结束。

        Args:
            query (str): 用户查询。
            history(Optional[Sequence[BaseMessage]]): 聊天历史
            memory_query(str): 记忆检索的查询
            record(bool): 主观控制是否记录记忆

        Yields:
            AgentResult: 流式返回的结果片段。中间结果通常包含增量生成的回复内容；
                最后一次返回的结果包含``done=True``、``usage`` 和 ``history`` 信息。
        '''
        async for item in super().ainvoke_stream(query, history=history, memory_query=memory_query, record=record, **kwargs):
            yield item

    def _handle_input(self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, memory_query: Optional[str] = None, record: bool = True, **kwargs) -> List[BaseMessage]:
        '''
        将输入的用户查询转为输入messages的模块

        Args:
            query(str): 用户查询
            history(Optional[Sequence[BaseMessage]]): 聊天历史
            memory_query(str): 记忆检索的查询
            record(bool): 主观控制是否记录记忆

        Returns:
            List[BaseMessage]: 智能体的输入messages
        '''
        messages = []
        inner_query = memory_query if memory_query is not None else query
        if self.memory_enable:
            memories =[]
            if self.memory_retrieve_func:
                memories = self.memory_retrieve_func(inner_query)
                if memories:
                    memory_template = Template(MEMORY_MESSAGE_TEMPLATE)
                    memory_message = SystemMessage(content=memory_template.render(memories=memories))
                    messages.append(memory_message)
                    # 此处将长期记忆设置为系统角色，在后续的simplify_history中会被过滤，故无需额外处理

            if self.memory_record_func and record:
                loop = asyncio.get_running_loop()
                loop.run_in_executor(self.memory_thread_pool, self.memory_record_func, inner_query, memories)

        ctx_messages = super()._handle_input(query=query, history=history, **kwargs)
        messages.extend(ctx_messages)

        return messages

    async def _ahandle_input(self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, memory_query: Optional[str] = None, record: bool = True, **kwargs) -> List[BaseMessage]:
        '''
        _handle_input 的 异步实现
        '''
        messages = []
        inner_query = memory_query if memory_query is not None else query
        if self.memory_enable:
            memories = []
            if self.memory_retrieve_afunc:
                memories = await self.memory_retrieve_afunc(inner_query)
                if memories:
                    memory_template = Template(MEMORY_MESSAGE_TEMPLATE)
                    memory_message = SystemMessage(content=memory_template.render(memories=memories))
                    messages.append(memory_message)
                    # 此处将长期记忆设置为系统角色，在后续的simplify_history中会被过滤，故无需额外处理

            if self.memory_record_afunc and record:
                asyncio.create_task(self.memory_record_afunc(inner_query, memories))
        
        ctx_messages = await super()._ahandle_input(query=query, history=history, **kwargs)
        messages.extend(ctx_messages)

        return messages
