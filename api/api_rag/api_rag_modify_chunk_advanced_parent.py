'''
    编辑分段 - 高级分段 - 父分段

    根据段落内容，生成向量
    修改表knowledge_base_document中对应记录的preview_chunks、word_count列
    移除表knowledge_base_doc_vector中原父段下的所有子段
    增加记录至表knowledge_base_doc_vector
'''

from typing import Dict
import traceback
import json
import asyncio
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from fastapi import Request, Body

from api.api import app, get_api_client_tag
from core.database.database_factory import DatabaseFactory
from core.rag.vectorizer.vectorize_processor import VectorizeProcessor
from core.rag.utils.sql_expression import (
    SQL_EXPRESSION_DOCUMENT_UPDATE_CURD, 
    SQL_EXPRESSION_DOCUMENT_SELECT_CURD,
    SQL_EXPRESSION_VECTOR_INSERT, 
    SQL_EXPRESSION_VECTOR_DELETE_CURD, 
)
from core.rag.utils.rag_utils import (
    len_without_link, 
    list_to_pgvector_str, 
    generate_preview_chunks_from_documents,
)
from core.rag.splitter.splitter_entities import SplitType
from api.api_rag.entities import ChunkStatus
from core.rag.entities.document import Document
from core.rag.splitter.splitter_fulldoc import FulldocSplitter
from config.config import get_config
from logger import get_logger

logger = get_logger('api')

class ApiRagRequestModel(BaseModel):
    knowledge_base_id: int = Field(..., description='知识库id')
    document_id: int = Field(..., description='文档id')
    parent_idx: int = Field(..., description='父段序号')
    chunk_content: str = Field(..., description='父段内容')
    sonchunk_setting: Dict = Field(..., description='子段配置')
    model_instance_provider: str = Field(..., description='Embedding模型加载方式')
    model_instance_config: Dict = Field(..., description='Embedding模型配置信息')

    @model_validator(mode='after')
    def validate_input(self) -> 'ApiRagRequestModel':
        if not self.chunk_content:
            raise ValueError('Illegal chunk_content.')
        return self
    
class ApiRagResponseModel(BaseModel):
    code: int = Field(..., description='状态码，成功为1000，失败为2000')
    msg: str = Field(..., description='状态信息')

@app.post(
    path='/Voicecomm/VoiceSageX/Rag/ModifyChunkAdvancedParent', 
    response_model=ApiRagResponseModel,
    response_model_exclude_none=False, 
)
async def handler(conn: Request, body: Dict = Body(...)):
    tag, task_id = get_api_client_tag(conn)
    logger.debug(f'{task_id} {tag}')

    # 格式校验
    try:
        logger.debug(f'{task_id} Request body:\n{json.dumps(body, indent=4, ensure_ascii=False)}')
        request = ApiRagRequestModel.model_validate(body)
    except Exception as e:
        logger.error(f'{task_id} Fail to validate pydantic instance for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: The request body field is incorrect for:{type(e).__name__}: {str(e)}.'
        )
    
    # 获取数据库实例
    try:
        embedding_batch_size = get_config().get('server', {}).get('embedding_batch_size', 8)

        database_info = get_config()['dependent_info']['database']
        db = DatabaseFactory.get_database(database_info['type'])

        document_path_prefix = get_config().get('dependent_info').get('knowledge_base').get('document_path_prefix')

        vectorizer = VectorizeProcessor().get_vectorizer(
            model_instance_provider=request.model_instance_provider, 
            model_instance_config=request.model_instance_config, 
        )

        logger.debug(f'{task_id} Got required instance.')
    except Exception as e:
        logger.error(f'{task_id} Fail to get required instance for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to get required instance for:{type(e).__name__}: {str(e)}.'
        )
    
    # 处理数据
    try:
        # 父段切分
        splitter = FulldocSplitter(
            length_function=len_without_link, sonchunk_setting=request.sonchunk_setting
        )
        parent_documents = await splitter.split_chunks([
            Document(page_content=request.chunk_content, metadata={"source": "Additional chunk"})
        ])
        parent_document = parent_documents[0]
        logger.debug(f'{task_id} parent page_content: {parent_document.page_content}')
        logger.debug(f'{task_id} number of child chunks: {len(parent_document.children)}')

        # 生成向量，与记录
        failed_count = 0
        chunk_records = []
        for i in range(0, len(parent_document.children), embedding_batch_size):
            batch_documents = parent_document.children[i: i + embedding_batch_size]
            
            coros = [VectorizeProcessor.vectorize(
                vectorizer, [document.page_content]
            ) for document in batch_documents]

            outputs = await asyncio.gather(*coros, return_exceptions=True)
            for idx, output in enumerate(outputs):
                if isinstance(output, Exception):
                    failed_count += 1
                    batch_documents[idx].metadata['failed_reason'] = f"{type(output).__name__}:{str(output)[:100]}"
                    chunk_records.append(
                        [
                            batch_documents[idx].metadata.get('content_id'),
                            request.knowledge_base_id,
                            request.document_id,
                            batch_documents[idx].page_content,
                            parent_document.page_content,
                            'metadata', # json.dumps(batch_documents[idx].metadata),    # 此处先不生成json str
                            None,
                            None,
                            ChunkStatus.FAILED.value,
                            batch_documents[idx].metadata.get('content_hash'),
                            SplitType.ADVANCED_PARAGRAPH.value, # 先随便设置一个值
                        ]
                    )
                    logger.warning(f'{task_id} [{i + idx}] Vectorize failed for:\n{repr(output)}')
                else:
                    chunk_records.append(
                        [
                            batch_documents[idx].metadata.get('content_id'),
                            request.knowledge_base_id,
                            request.document_id,
                            batch_documents[idx].page_content,
                            parent_document.page_content,
                            'metadata', # json.dumps(batch_documents[idx].metadata),    # 此处先不生成json str
                            list_to_pgvector_str(output[0]['vector']),
                            json.dumps(output[0]['usage']),
                            ChunkStatus.SUCCESS.value,
                            batch_documents[idx].metadata.get('content_hash'),
                            SplitType.ADVANCED_PARAGRAPH.value, # 先随便设置一个值
                        ]
                    )

        child_chunks_num = len(parent_document.children)
        if failed_count == child_chunks_num:
            raise RuntimeError(f"All chunks failed when vectorize.")
    except Exception as e:
        logger.error(f'{task_id} Fail to process data for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to process data for:{type(e).__name__}: {str(e)}.'
        )

    # 事务
    try:
        async with db.conn() as conn:
            async with conn.transaction():
                # 从表 knowledge_base_document 中读取文档的记录
                document_select_sql = SQL_EXPRESSION_DOCUMENT_SELECT_CURD.format(
                    ','.join(['preview_chunks', 'word_count', 'chunking_strategy', 'name', 'unique_name'])
                )
                rows = await conn.fetch(document_select_sql, request.document_id)
                if not rows or len(rows) > 1:
                    raise ValueError(f"No records or not unique.")
                record = dict(rows[0])
                record['preview_chunks'] = json.loads(record['preview_chunks'])
                logger.debug(f'{task_id} Select record from "knowledge_base_document".')
                logger.debug(f'{task_id} chunking_strategy: {record["chunking_strategy"]}')
                logger.debug(f'{task_id} name: {record["name"]}')
                logger.debug(f'{task_id} unique_name: {record["unique_name"]}')

                # 在preview_chunks中找到对应的父段
                target_chunk = None
                for chunk in record['preview_chunks']:
                    if chunk['id'] == request.parent_idx:
                        target_chunk = chunk
                        logger.debug(f'{task_id} Original:\n{json.dumps(target_chunk, ensure_ascii=False, indent=4)}')
                        break

                if not target_chunk:
                    raise ValueError("chunk not found.")
                
                logger.debug(f'{task_id} Locate to the specified parent chunk, index [{request.parent_idx}].')

                # 增加记录至表knowledge_base_doc_vector，并更新metadata
                for idx, chunk_record in enumerate(chunk_records):
                    # 更新metadata
                    parent_document.children[idx].metadata['source'] = str(Path(document_path_prefix) / 'knowledge-base/documents/' / record['unique_name'])
                    parent_document.children[idx].metadata['title'] = record['name']
                    # 更新chunk_record中的metadata
                    chunk_record[5] = json.dumps(parent_document.children[idx].metadata)

                    # 更新切分策略
                    chunk_record[-1] = record['chunking_strategy']
                    id = await conn.fetchrow(SQL_EXPRESSION_VECTOR_INSERT, *chunk_record)
                    parent_document.children[idx].metadata['primary_key'] = id['id']
                    logger.debug(f'{task_id} inserted chunk record: {id["id"]}.')

                # 生成新增chunk的preview_chunks
                new_preview_chunks = generate_preview_chunks_from_documents(
                    SplitType.ADVANCED_PARAGRAPH.value,
                    [parent_document],
                    is_edited=True,
                )
                new_preview_chunk = new_preview_chunks[0]
                logger.debug(f'{task_id} new preview_chunk:\n{json.dumps(new_preview_chunk["content"], ensure_ascii=False, indent=4)}')

                # 记录当前父段的所有子段主键
                child_pks = []
                for child_chunk in target_chunk['content']:
                    child_pks.append(child_chunk['primary_key'])
                logger.debug(f'{task_id} primary keys need to be deleted:\n{child_pks}.')

                # 修改该记录的preview_chunks、word_count列
                logger.debug(f'{task_id} original word count: {record["word_count"]}.')
                logger.debug(f'{task_id} original chunk character: {target_chunk["character"]}.')

                record['word_count'] -= target_chunk['character']
                record['word_count'] += parent_document.metadata['content_len']

                logger.debug(f'{task_id} original chunk character: {parent_document.metadata["content_len"]}.')
                logger.debug(f'{task_id} new word count: {record["word_count"]}.')

                target_chunk['content'] = new_preview_chunk['content']
                target_chunk['character'] = new_preview_chunk['character']
                target_chunk['isEdited'] = True

                logger.debug(f'{task_id} new content: {new_preview_chunk["content"]}')

                # 修改表knowledge_base_document中对应记录的preview_chunks、word_count列
                document_update_sql = SQL_EXPRESSION_DOCUMENT_UPDATE_CURD.format(
                    ','.join(["preview_chunks=$2::jsonb", "word_count=$3"])
                )
                await conn.execute(
                    document_update_sql, 
                    request.document_id, 
                    json.dumps(record['preview_chunks']),
                    record['word_count'],
                )
                logger.debug(f'{task_id} Update "knowledge_base_document".')

                # 移除表knowledge_base_doc_vector中原父段下的所有子段
                await conn.execute(
                    SQL_EXPRESSION_VECTOR_DELETE_CURD, 
                    child_pks,
                )
                logger.debug(f'{task_id} Delete record from "knowledge_base_doc_vector".')
                
    except Exception as e:
        logger.error(f'{task_id} Fail to excute sql transaction for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to excute sql transaction for:{type(e).__name__}: {str(e)}.'
        )

    # 响应
    response = ApiRagResponseModel(
        code=1000, 
        msg=f'{task_id} Success.'
    )
    logger.debug(f'{task_id} Response:\n{response.model_dump_json(indent=4)}')

    return response