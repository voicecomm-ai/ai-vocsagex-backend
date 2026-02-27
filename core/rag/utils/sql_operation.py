from typing import List, Dict, Tuple, Optional
import json
import traceback

from core.database.database_factory import DatabaseFactory
from core.rag.entities.document import Document
from core.rag.utils.sql_expression import (
    SQL_EXPRESSION_DOCUMENT_UPDATE_STATUS_BY_ID, 
    SQL_EXPRESSION_DOCUMENT_UPDATE_STATUS_BY_KNOWLEDGE_BASE_ID, 
    SQL_EXPRESSION_DOCUMENT_UPDATE,
    SQL_EXPRESSION_VECTOR_INSERT,
    SQL_EXPRESSION_VECTOR_SELECT,
    SQL_EXPRESSION_VECTOR_UPDATE,
    SQL_EXPRESSION_VECTOR_DELETE_BY_DOCUMENT_ID, 
)
from logger import get_logger

logger = get_logger('sql')

async def modify_document_status_by_id(
    db, 
    key_id: int,
    process_status: str
) -> str:
    row = await db.fetchrow(
        SQL_EXPRESSION_DOCUMENT_UPDATE_STATUS_BY_ID,
        key_id,
        process_status,
    )
    if row:
        return row['name']
    else:
        raise RuntimeError(f'Cannot find "{key_id}" in knowledge_base_document')

async def modify_document_status_by_knowledge_base_id(
    db, 
    knowledge_base_id: int,
    process_status: str
):
    await db.execute(
        SQL_EXPRESSION_DOCUMENT_UPDATE_STATUS_BY_KNOWLEDGE_BASE_ID,
        knowledge_base_id,
        process_status,
    )

async def update_documents_into_db(
    db,
    key_id: int,
    split_type: str,
    preview_chunks: List[Dict],
    process_status: str,
    word_count: int,
    process_failed_reason: str,
):
    '''
        分段后，将分段信息写入数据表knowledge_base_document
    '''
    # 查询表knowledge_base_document中是否有id为key_id的行
    # 若有，则想id=key_id的chunking_strategy、preview_chunks、chunks列中写入数据

    rows = await db.fetchrow(
        SQL_EXPRESSION_DOCUMENT_UPDATE,
        # split_type,
        json.dumps(preview_chunks),
        process_status,
        key_id,
        word_count,
        process_failed_reason,
    )

    if rows:
        pass
    else:
        raise RuntimeError(f'Cannot find "{key_id}" in knowledge_base_document')


async def insert_vector_into_db(db, records: List[Tuple]):
    '''
        保存并处理后，将向量信息写入数据表knowledge_base_doc_vector
    '''
    ids = []
    for record in records:
        id = await db.fetchrow(SQL_EXPRESSION_VECTOR_INSERT, *record)
        ids.append(id)

    return ids


async def select_vector_from_db(db, knowledge_base_id: int):
    '''
        重建向量前，从数据表knowledge_base_doc_vector中查询记录，并修改每条记录的处理状态
    '''
    return await db.fetch(SQL_EXPRESSION_VECTOR_SELECT, knowledge_base_id)


async def update_vector_into_db(db, records: List[Tuple]):
    '''
        重建向量后，修改select_vector_from_db返回记录的处理状态、向量、tokens用量
    '''
    await db.executemany(SQL_EXPRESSION_VECTOR_UPDATE, records)


async def select_vector_by_knowledge_base_id(
    db, 
    need_vector: bool, 
    knowledge_base_id: int, 
    top_k: int, 
    query_vector: str, 
    score_threshold: Optional[float] = None, 
    metadata_condition: Optional[str] = None,
) -> List[Document]:
    '''
        向量检索
    '''
    sql = build_vector_select_sql(need_vector, metadata_condition)
    logger.debug(f"SQL:\n{sql}")

    rows = await db.fetch(
        sql,
        knowledge_base_id, 
        top_k, 
        query_vector
    )

    documents = []
    for row in rows:
        if score_threshold and row["score"] < score_threshold:
            continue

        try:
            documents.append(
                Document(
                    page_content=row["retrieve_content"],
                    vector=json.loads(row["vector"]) if need_vector else None,
                    metadata=json.loads(row["metadata"]),
                )
            )
            documents[-1].metadata["score"] = row["score"] if row["score"] >= 0 else 0.0
            documents[-1].metadata["context_content"] = row["context_content"]
            documents[-1].metadata["document_id"] = row["document_id"]
            documents[-1].metadata["knowledge_base_id"] = knowledge_base_id
        except Exception as e:
            logger.warning(f'Ignore record: {row} for:\n{traceback.format_exc()}')
            continue

    return documents


def build_vector_select_sql(need_vector: bool, metadata_condition: Optional[str] = None,) -> str:
    base_fields = [
        "document_id",
        "retrieve_content",
        "context_content",
        "metadata",
    ]

    # 可选字段：vector + 向量距离
    if need_vector:
        base_fields.append("vector")
    
    metadata_condition = metadata_condition or '1=1'

    # 向量相似度字段
    base_fields.append("1 - (vector <=> $3::vector) AS score")

    fields_clause = ", ".join(base_fields)

    return f"""
    SELECT {fields_clause}
    FROM knowledge_base_doc_vector
    WHERE knowledge_base_id = $1
          AND process_status = 'SUCCESS'
          AND status = 'ENABLE'
          AND document_id IN (
              SELECT id
              FROM knowledge_base_document as d
              WHERE knowledge_base_id = $1
                    AND status = 'ENABLE'
                    AND is_archived = FALSE
                    AND {metadata_condition}
          )
    ORDER BY score DESC
    LIMIT $2 
    """

async def full_text_search_by_knowledge_base_id(
    db, 
    need_vector: bool, 
    knowledge_base_id: int, 
    top_k: int, 
    query_text: str, 
    score_threshold: Optional[float] = None, 
    metadata_condition: Optional[str] = None,
) -> List[Document]:
    '''
        全文检索
    '''
    sql = build_full_text_select_sql(need_vector, metadata_condition)
    logger.debug(f"SQL:\n{sql}")

    rows = await db.fetch(
        sql, knowledge_base_id, top_k, query_text
    )

    documents = []
    for row in rows:
        if score_threshold and row["score"] < score_threshold:
            continue
        try:
            documents.append(
                Document(
                    page_content=row["retrieve_content"],
                    vector=json.loads(row["vector"]) if need_vector else None,
                    metadata=json.loads(row["metadata"]),
                )
            )
            documents[-1].metadata["score"] = row["score"] if row["score"] >= 0 else 0.0
            documents[-1].metadata["context_content"] = row["context_content"]
            documents[-1].metadata["document_id"] = row["document_id"]
            documents[-1].metadata["knowledge_base_id"] = knowledge_base_id
        except Exception as e:
            logger.warning(f'Ignore record: {row} for:\n{traceback.format_exc()}')
            continue

    return documents

def build_full_text_select_sql(need_vector: bool, metadata_condition: Optional[str] = None,) -> str:
    base_fields = [
        "document_id",
        "retrieve_content",
        "context_content",
        "metadata",
        "bigm_similarity(unistr($3), coalesce(retrieve_content, '')) AS score",
    ]
    # 可选字段：vector + 向量距离
    if need_vector:
        base_fields.append("vector")

    metadata_condition = metadata_condition or '1=1'

    return f"""
    SELECT {', '.join(base_fields)}
    FROM knowledge_base_doc_vector
    WHERE knowledge_base_id = $1
          AND process_status = 'SUCCESS'
          AND status = 'ENABLE'
          AND document_id IN (
              SELECT id
              FROM knowledge_base_document as d
              WHERE knowledge_base_id = $1
                    AND status = 'ENABLE'
                    AND is_archived = FALSE
                    AND {metadata_condition}
          )
    ORDER BY score DESC
    LIMIT $2
    """

async def delete_vector_by_document_id(db, document_id):
    await db.execute(
        SQL_EXPRESSION_VECTOR_DELETE_BY_DOCUMENT_ID, 
        document_id
    )