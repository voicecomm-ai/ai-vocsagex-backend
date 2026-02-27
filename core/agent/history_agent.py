from typing import Optional, List, Sequence, Generator, AsyncGenerator, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import (
    UsageMetadata, 
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
)
from jinja2 import Template

from core.agent.base_agent import BaseAgent
from core.agent.base_model import AgentResult
from core.agent.base_constants import (
    HISTORY_SUMMARY_SYSTEM_PROMPT, 
    HISTORY_SUMMARY_HUMAN_PROMPT, 
    HISTORY_SUMMARY_MESSAGE_TEMPLATE, 
)
from core.agent.base_utils import (
    get_history_turn, 
    truncate_history, 
    estimate_tokens, 
    simplify_history, 
)


class HistoryAgent(BaseAgent):
    '''
        含有聊天历史的 ReAct 智能体
    '''
    def __init__(
        self, name: str, description: str, recursion_limit: int = 10, 
        depth: int = 10, summarize: bool = False, summarize_limit: int = 2048, 
        summarize_model: Optional[BaseChatModel] = None, 
        **kwargs
    ):
        '''
        创建含有聊天历史的ReAct智能体

        Args:
            name(str): 智能体名称
            description(str): 智能体描述
            recursion_limit(int): 内部迭代次数
            depth(int): 聊天历史最大轮次
                        0:  不记录历史记录
                        -1: 不限制轮次
                        正数：限制轮次
                        其他：设置为0
            summarize(bool): 是否对聊天历史生成摘要
            summarize_limit(int): 聊天历史token限制
            summarize_model(Optional[BaseChatModel]): 生成摘要的聊天模型

        Notes:
            - 当`depth=-1`且`summarize=False`时，将退化为带有上下文的`BaseAgent`
            - 启用summarize时，可以传入外部的summarize_model，若不传将使用智能体本身的聊天模型
            - 启用summarize时，当聊天历史轮次超过`depth`或token超过`summarize_limit`，均会生成摘要
            - 启用summarize时，将产生额外的token用量
            - 生成摘要时，会将聊天分为远、近两部分，近的聊天历史最多保留3轮，对远的进行摘要，近的进行保留
            - 若近的聊天历史超过`summarize_limit`限制，则对所有聊天历史进行摘要
            - `invoke`系列方法生成结果中的history未截断，截断均发生在agent调用前
        '''
        depth = round(depth)
        self.history_depth = 0 if depth < -1 else depth
        self.history_summarize = summarize
        self.history_summarize_limit = summarize_limit
        self.history_summarize_model = summarize_model

        super().__init__(name, description, recursion_limit, **kwargs)

    def compile(self):
        if self.history_summarize_model is None and self.history_summarize:
            self.history_summarize_model = self.chat_model

        super().compile()

    def invoke(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, **kwargs
    ) -> AgentResult:
        '''
        以同步、非流的方式进行智能体调用

        Args:
            query(str): 用户查询
            history(Optional[Sequence[BaseMessage]]): 聊天历史

        Returns:
            AgentResult: 智能体响应结果
        '''
        return super().invoke(query, history=history, **kwargs)

    async def ainvoke(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, **kwargs
    ) -> AgentResult:
        '''
        以异步、非流的方式进行智能体调用

        Args:
            query(str): 用户查询
            history(Optional[Sequence[BaseMessage]]): 聊天历史

        Returns:
            AgentResult: 智能体响应结果
        '''
        return await super().ainvoke(query, history=history, **kwargs)

    def invoke_stream(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, **kwargs
    ) -> Generator[AgentResult, None, None]:
        '''
        以同步、流的方式进行智能体调用

        该方法会逐步返回模型生成的结果片段，直到流结束。

        Args:
            query(str): 用户查询
            history(Optional[Sequence[BaseMessage]]): 聊天历史

        Yields:
            AgentResult: 流式返回的结果片段。中间结果通常包含增量生成的回复内容；
                最后一次返回的结果包含``done=True``、``usage`` 和 ``history`` 信息。
        '''
        yield from super().invoke_stream(query, history=history, **kwargs)

    async def ainvoke_stream(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, **kwargs
    ) -> AsyncGenerator[AgentResult, None]:
        '''
        以异步、流式的方式进行智能体调用。

        该方法会在异步上下文中逐步返回模型生成的结果片段，直到流结束。

        Args:
            query (str): 用户查询。
            history(Optional[Sequence[BaseMessage]]): 聊天历史

        Yields:
            AgentResult: 流式返回的结果片段。中间结果通常包含增量生成的回复内容；
                最后一次返回的结果包含``done=True``、``usage`` 和 ``history`` 信息。
        '''
        async for item in super().ainvoke_stream(query, history=history, **kwargs):
            yield item

    def _handle_input(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, **kwargs
    ) -> List[BaseMessage]:
        '''
        将输入的用户查询转为输入messages的模块

        Args:
            query(str): 用户查询
            history(Optional[Sequence[BaseMessage]]): 聊天历史

        Returns:
            List[BaseMessage]: 智能体的输入messages
        '''
        if history is None:
            history = []

        if self.history_depth == 0:
            return super()._handle_input(query, **kwargs)

        messages = []

        if (
            self.history_summarize
            and (
                estimate_tokens(history) >= self.history_summarize_limit
                or
                (
                    self.history_depth != -1
                    and 
                    get_history_turn(history) > self.history_depth
                )
            )
        ):
            # 生成 history 摘要

            # 保留的近期历史轮次
            recent_turn_num = max(self.history_depth // 2, 3)
        
            recent_history = truncate_history(history, recent_turn_num)
            old_history = history[:len(history) - len(recent_history)]

            if estimate_tokens(recent_history) >= self.history_summarize_limit:
                history_summary, history_summary_usage = self._generate_history_summary(history)
                history_summary_template = Template(HISTORY_SUMMARY_MESSAGE_TEMPLATE)
                history_summary_message = AIMessage(
                    content=history_summary_template.render(history_summary=history_summary), 
                    usage_metadata=history_summary_usage, 
                )
                messages.append(history_summary_message)
            else:
                history_summary, history_summary_usage = self._generate_history_summary(old_history)
                history_summary_template = Template(HISTORY_SUMMARY_MESSAGE_TEMPLATE)
                history_summary_message = AIMessage(
                    content=history_summary_template.render(history_summary=history_summary), 
                    usage_metadata=history_summary_usage, 
                )
                messages.append(history_summary_message)
                messages.extend(simplify_history(recent_history))
        else:
            if self.history_depth != -1 and get_history_turn(history) > self.history_depth:
                messages.extend(truncate_history(history, self.history_depth))
            else:
                messages.extend(history)
            # 清洗传入的history，避免传入无用信息
            messages = simplify_history(messages)

        query_messages = super()._handle_input(query, **kwargs)
        messages.extend(query_messages)

        return messages

    async def _ahandle_input(
        self, query: str, *, history: Optional[Sequence[BaseMessage]] = None, **kwargs
    ) -> List[BaseMessage]:
        '''
        _handle_input 的 异步实现
        '''
        if history is None:
            history = []

        if self.history_depth == 0:
            return super()._handle_input(query, **kwargs)

        messages = []

        if (
            self.history_summarize
            and (
                estimate_tokens(history) >= self.history_summarize_limit
                or
                (
                    self.history_depth != -1
                    and 
                    get_history_turn(history) > self.history_depth
                )
            )
        ):
            # 生成 history 摘要
            
            # 保留的近期历史轮次
            recent_turn_num = self.history_depth // 2
            recent_turn_num = recent_turn_num if recent_turn_num <= 3 else 3
        
            recent_history = truncate_history(history, recent_turn_num)
            old_history = history[:len(history) - len(recent_history)]

            if estimate_tokens(recent_history) >= self.history_summarize_limit:
                history_summary, history_summary_usage = await self._agenerate_history_summary(history)
                history_summary_template = Template(HISTORY_SUMMARY_MESSAGE_TEMPLATE)
                history_summary_message = AIMessage(
                    content=history_summary_template.render(history_summary=history_summary), 
                    usage_metadata=history_summary_usage, 
                )
                messages.append(history_summary_message)
            else:
                history_summary, history_summary_usage = await self._agenerate_history_summary(old_history)
                history_summary_template = Template(HISTORY_SUMMARY_MESSAGE_TEMPLATE)
                history_summary_message = AIMessage(
                    content=history_summary_template.render(history_summary=history_summary), 
                    usage_metadata=history_summary_usage, 
                )
                messages.append(history_summary_message)
                messages.extend(simplify_history(recent_history))
        else:
            if self.history_depth != -1 and get_history_turn(history) > self.history_depth:
                messages.extend(truncate_history(history, self.history_depth))
            else:
                messages.extend(history)
            # 清洗传入的history，避免传入无用信息
            messages = simplify_history(messages)

        query_messages = super()._handle_input(query, **kwargs)
        messages.extend(query_messages)

        return messages

    def _generate_history_summary(
        self, history: Sequence[BaseMessage]
    ) -> Tuple[str, UsageMetadata]:
        '''
        生成聊天历史的上下文摘要

        Args:
            history(Sequence[BaseMessage]): 聊天历史
        
        Returns:
            tuple: 一个二元组，包含：
            - str: 上下文摘要
            - UsageMetadata: token用量
        '''
        messages = []
        messages.append(SystemMessage(content=HISTORY_SUMMARY_SYSTEM_PROMPT))
        messages.extend(history)
        messages.append(HumanMessage(content=HISTORY_SUMMARY_HUMAN_PROMPT))

        response = self.history_summarize_model.invoke(messages)

        return response.content.strip(), response.usage_metadata

    async def _agenerate_history_summary(
        self, history: Sequence[BaseMessage]
    ) -> Tuple[str, UsageMetadata]:
        '''
        _generate_history_summary 的异步实现
        '''
        messages = []
        messages.append(SystemMessage(content=HISTORY_SUMMARY_SYSTEM_PROMPT))
        messages.extend(history)
        messages.append(HumanMessage(content=HISTORY_SUMMARY_HUMAN_PROMPT))

        response = await self.history_summarize_model.ainvoke(messages)

        return response.content.strip(), response.usage_metadata
