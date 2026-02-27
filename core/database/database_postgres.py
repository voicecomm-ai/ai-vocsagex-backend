from typing import List, Tuple, Any, Optional

import asyncpg

from core.database.database_base import BaseDatabase

class PostgresDatabase(BaseDatabase):
    def __init__(self, **kwargs):
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(
        self,
        user: str,
        password: str,
        database: str,
        host: str = "localhost",
        port: int = 5432,
        min_size: int = 1,
        max_size: int = 10,
        **kwargs
    ):
        self._pool = await asyncpg.create_pool(
            user=user,
            password=password,
            database=database,
            host=host,
            port=port,
            min_size=min_size,
            max_size=max_size,
        )

    async def disconnect(self):
        if self._pool:
            await self._pool.close()

    async def fetch(self, query: str, *args):
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def execute(self, query: str, *args):
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def executemany(self, query: str, records: list[tuple]):
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(query, records)

    async def transaction(self, operations: List[Tuple[str, Tuple]]):
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for sql, params in operations:
                    await conn.execute(sql, *params)

    def conn(self):
        return self._pool.acquire()