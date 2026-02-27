from typing import List, Optional

from core.memory.constants import (
    MEMORY_RETRIEVE_SQL, 
    MEMORY_INSERT_SQL, 
    MEMORY_DELETE_SQL, 
    MEMORY_UPDATE_SQL, 
)

def list_to_pgvector_str(vec: List[float]) -> str:
    return '[' + ','.join(f'{v}' for v in vec) + ']'

async def select_memory(
    db, 
    vector: List[float], 
    application_id: int, 
    user_id: int, 
    agent_id: int,
    expired_time: Optional[str],
    data_type: str,
):
    rows = await db.fetch(
        MEMORY_RETRIEVE_SQL,
        list_to_pgvector_str(vector),
        application_id,
        user_id,
        agent_id,
        data_type,
        expired_time,
    )

    result = []
    for row in rows:
        result.append(
            {
                "id": row["id"],
                "content": row["content"],
                "score": row["score"],
            }
        )

    return result

async def insert_memory(
    db,
    content: str,
    vector: List[float], 
    application_id: int, 
    user_id: int, 
    agent_id: int,
    data_type: str,
) -> int:
    row = await db.fetchrow(
        MEMORY_INSERT_SQL, 
        content, 
        list_to_pgvector_str(vector),
        application_id,
        user_id,
        agent_id,
        data_type,
    )

    return row["id"]

async def delete_memory(
    db, 
    ids: List[int], 
    application_id: int, 
    user_id: int, 
    agent_id: int, 
):
    await db.execute(
        MEMORY_DELETE_SQL, 
        ids, 
        application_id,
        user_id,
        agent_id,
    )

async def update_memory(
    db, 
    id : int,
    content: str,
    vector: List[float], 
):
    await db.execute(
        MEMORY_UPDATE_SQL, 
        id, 
        content, 
        list_to_pgvector_str(vector),
    )