from abc import ABC, abstractmethod
from typing import List, Tuple, Any

class BaseDatabase(ABC):
    @abstractmethod
    async def connect(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self):
        raise NotImplementedError

    @abstractmethod
    async def fetch(self, query: str, *args):
        raise NotImplementedError

    @abstractmethod
    async def fetchrow(self, query: str, *args) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, query: str, *args):
        raise NotImplementedError

    @abstractmethod
    async def executemany(self, query: str, records: List[Tuple]):
        raise NotImplementedError
    
    @abstractmethod
    async def transaction(self, operations: List[Tuple[str, Tuple]]):
        '''
            支持简单的事务
            如果复杂事务，请使用conn()方法自行获取连接的异步上下文管理器
        '''
        raise NotImplementedError
    
    @abstractmethod
    def conn(self):
        '''
            获取连接的异步上下文管理器

            async with db.conn() as conn:
                ......

        '''
        raise NotImplementedError