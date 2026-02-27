from abc import ABC, abstractmethod
from typing import Generator, AsyncGenerator

from core.agent.base_model  import AgentResult

class BaseMultiAgent(ABC):

    @abstractmethod
    def invoke(self, query: str, **kwargs) -> AgentResult:
        raise NotImplementedError
    
    @abstractmethod
    async def ainvoke(self, query: str, **kwargs) -> AgentResult:
        raise NotImplementedError

    @abstractmethod
    def invoke_stream(self, query: str, **kwargs) -> Generator[AgentResult, None, None]:
        raise NotImplementedError

    @abstractmethod
    async def ainvoke_stream(self, query: str, **kwargs) -> AsyncGenerator[AgentResult, None]:
        raise NotImplementedError
    
