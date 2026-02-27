'''
    保存并处理：普通分段处理
'''

from typing import Dict, Optional
from enum import Enum
import traceback
from uuid import uuid4
import asyncio
import json
from threading import Event
import concurrent.futures
from pathlib import Path

from pydantic import Field
from fastapi import Request, Body

from api.api import app, get_api_client_tag
from api.api_rag.api_rag_preview_chunk_normal import ApiRagRequestModel as RequestModel
from api.api_rag.api_rag_preview_chunk_normal import ApiRagResponseModel
from core.rag.extractor.extract_processor import ExtractProcessor
from core.rag.cleaner.clean_processor import CleanProcessor
from core.rag.splitter.split_processor import SplitProcessor
from core.rag.splitter.splitter_entities import SplitType
from core.rag.vectorizer.vectorize_processor import VectorizeProcessor
from core.rag.utils.sql_operation import (
    update_documents_into_db, 
    insert_vector_into_db,
    modify_document_status_by_id,
    delete_vector_by_document_id, 
)
from core.rag.utils.rag_utils import (
    generate_preview_chunks_from_documents, 
    generate_chunks_from_documents, 
    list_to_pgvector_str, 
    len_without_link, 
)
from core.database.database_factory import DatabaseFactory
from api.api_rag.entities import DocumentStatus, ChunkStatus
from logger import get_logger
from config.config import get_config

logger = get_logger('api')

class ApiRagRequestModel(RequestModel):
    knowledge_base_id: int = Field(..., description='java内部知识库id')
    model_instance_provider: str = Field(..., description='模型供应商')
    model_instance_config: Dict = Field(..., description='模型配置信息')

async def gather_coros(coros):
    return await asyncio.gather(*coros, return_exceptions=True)

async def async_task(task_id: str, stop_event: Event, request: ApiRagRequestModel) -> None:
    logger.debug(f'{task_id} Start processing.')

    # 获取数据库实例
    try:
        database_info = get_config()['dependent_info']['database']
        db = DatabaseFactory.get_database(database_info['type'])
    except asyncio.CancelledError:
        logger.warning(f'{task_id} Task cancel.')
        return
    except Exception as e:
        logger.error(f'{task_id} Fail to get database instance for:\n{traceback.format_exc()}')
        async with app.state.rag_tasks_mtx:
            app.state.rag_tasks_flag.pop(request.key_id, None)
        return

    # 修改文档状态为IN_PROGRESS
    try:
        name = await modify_document_status_by_id(
            db, 
            request.key_id, 
            DocumentStatus.IN_PROGRESS.value
        )
        logger.debug(f'{task_id} File outer name: {name}')
    except asyncio.CancelledError:
        logger.warning(f'{task_id} Task cancel.')
        return
    except Exception as e:
        logger.error(f'{task_id} Fail to modify document process_status for:\n{traceback.format_exc()}')
        async with app.state.rag_tasks_mtx:
            app.state.rag_tasks_flag.pop(request.key_id, None)
        return

    # 删除document下所有片段
    try:
        await delete_vector_by_document_id(db, request.key_id)
        logger.debug(f'{task_id} Delete vectors.')
    except asyncio.CancelledError:
        logger.warning(f'{task_id} Task cancel.')
        return
    except Exception as e:
        logger.error(f'{task_id} Fail to delete vectors for:\n{traceback.format_exc()}')
        async with app.state.rag_tasks_mtx:
            app.state.rag_tasks_flag.pop(request.key_id, None)
        return

    documents = []
    split_type = None
    preview_chunks = None
    chunks = None
    process_status = DocumentStatus.FAILED.value
    word_count = 0
    process_failed_reason = ""

    try:
        # 获取向量化实例
        vectorizer = VectorizeProcessor().get_vectorizer(**request.model_dump())
        logger.debug(f'{task_id} Got vectorizer.')

        # 提取
        document_path_prefix = get_config().get('dependent_info').get('knowledge_base').get('document_path_prefix')
        file_path = Path(document_path_prefix) / request.file_url
        file_id = str(uuid4())
        documents = ExtractProcessor().extract(
            file_path=file_path, 
            file_id=file_id
        )
        logger.debug(f'{task_id} Extracted.')

        # 清洗
        for document in documents:
            document.page_content = CleanProcessor.clean(
                document.page_content, 
                **(request.cleaner_setting.model_dump())
            )
        logger.debug(f'{task_id} Cleaned.')

        # 分段
        split_type, documents = await SplitProcessor.split(documents, task_id=task_id, stop_event=stop_event, **request.model_dump())
        logger.debug(f'{task_id} Splitted - {len(documents)} chunks.')

        # 为document添加序号
        for f_idx, f_document in enumerate(documents):
            f_document.metadata['idx'] = f_idx + 1
            if f_document.children:
                for s_idx, s_document in enumerate(f_document.children):
                    s_document.metadata['idx'] = s_idx + 1
                    s_document.metadata['f_idx'] = f_document.metadata['idx']

        # 原文档字符数为文档提取后的字符数，现修改为分段后的分段总长度
        word_count = sum(d.metadata['content_len'] for d in documents)
        if split_type == SplitType.NORMAL_QA.value:
            word_count += sum(d.metadata['answer_len'] for d in documents)
        logger.debug(f'{task_id} word count: {word_count}')

        process_status = DocumentStatus.SUCCESS.value
    except asyncio.CancelledError:
        logger.warning(f'{task_id} Task cancel.')
        return
    except Exception as e:
        logger.error(f'{task_id} Fail to generate chunks for:\n{traceback.format_exc()}')
        process_failed_reason = f"{type(e).__name__}:{str(e)[:100]}"

    if process_status == DocumentStatus.SUCCESS.value:
        # 向量化
        logger.debug(f'{task_id} Vectorizing...')
        embedding_batch_size = get_config().get('server', {}).get('embedding_batch_size', 8)
        failed_count = 0
        for i in range(0, len(documents), embedding_batch_size):

            batch_documents = documents[i: i + embedding_batch_size]

            coros = [VectorizeProcessor.vectorize(
                vectorizer, [document.page_content]
            ) for document in batch_documents]

            try:
                outputs = await gather_coros(coros)
            except asyncio.CancelledError:
                logger.warning(f'{task_id} Task cancel.')
                return

            values = []
            for idx, output in enumerate(outputs):
                
                # 在metadata中添加title字段
                batch_documents[idx].metadata['title'] = name

                if isinstance(output, Exception):
                    failed_count += 1

                    batch_documents[idx].metadata['failed_reason'] = f"{type(output).__name__}:{str(output)[:100]}"

                    values.append(
                        (
                            batch_documents[idx].metadata.get('content_id'),
                            request.knowledge_base_id,
                            request.key_id,
                            batch_documents[idx].page_content,
                            (
                                batch_documents[idx].page_content if split_type == SplitType.NORMAL.value 
                                else f'Question: {batch_documents[idx].page_content}\nAnswer: {batch_documents[idx].metadata.get("answer")}'
                            ),
                            json.dumps(batch_documents[idx].metadata),
                            None,
                            None,
                            ChunkStatus.FAILED.value,
                            batch_documents[idx].metadata.get('content_hash'),
                            split_type,
                        )
                    )
                    logger.warning(f'{task_id} [{i + idx}] Vectorize failed for:\n{repr(output)}')
                else:
                    # 成功
                    values.append(
                        (
                            batch_documents[idx].metadata.get('content_id'),
                            request.knowledge_base_id,
                            request.key_id,
                            batch_documents[idx].page_content,
                            (
                                batch_documents[idx].page_content if split_type == SplitType.NORMAL.value 
                                else f'Question: {batch_documents[idx].page_content}\nAnswer: {batch_documents[idx].metadata.get("answer")}'
                            ),
                            json.dumps(batch_documents[idx].metadata),
                            list_to_pgvector_str(output[0]['vector']),
                            json.dumps(output[0]['usage']),
                            ChunkStatus.SUCCESS.value,
                            batch_documents[idx].metadata.get('content_hash'),
                            split_type,
                        )
                    )
                    logger.debug(f'{task_id} [{i + idx}] Vectorize Done.')

            # 入向量库
            try:
                ids = await insert_vector_into_db(db, values)

                for id, idx in enumerate(range(i, i + len(batch_documents))):
                    documents[idx].metadata['primary_key'] = ids[id]['id']

                logger.debug(f'{task_id} document[{i}-{i + len(batch_documents) - 1}] inserted into database.')
            except asyncio.CancelledError:
                logger.warning(f'{task_id} Task cancel.')
                return
            except Exception:
                logger.error(f'{task_id} document[{i}-{i + len(batch_documents) - 1}] insert into database failed for:\n{traceback.format_exc()}')
        
        if len(documents) > 0 and failed_count == len(documents):
            process_failed_reason = 'RuntimeError:All chunks failed when vectorize.'
            process_status = DocumentStatus.FAILED.value

    try:
        preview_chunks = generate_preview_chunks_from_documents(split_type, documents, False)
    except asyncio.CancelledError:
        logger.warning(f'{task_id} Task cancel.')
        return
    except Exception as e:
        logger.error(f'{task_id} Fail to generate chunks json for:\n{traceback.format_exc()}')

    # 入文档库
    try:
        await update_documents_into_db(
            db,
            request.key_id,
            split_type,
            preview_chunks,
            process_status,
            word_count,
            process_failed_reason,
        )

        logger.debug(f'{task_id} Updated knowledge_base_document.')
    except asyncio.CancelledError:
        logger.warning(f'{task_id} Task cancel.')
        return
    except Exception:
        logger.error(f'{task_id} Fail to update knowledge_base_document for:\n{traceback.format_exc()}')

    async with app.state.rag_tasks_mtx:
        app.state.rag_tasks_flag.pop(request.key_id, None)

    logger.debug(f'{task_id} Finish processing.')


@app.post(path='/Voicecomm/VoiceSageX/Rag/ParseFileNormal', response_model=ApiRagResponseModel, response_model_exclude_none=False)
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
    
    # 任务导入线程池
    try:
        async with app.state.rag_tasks_mtx:
            if request.key_id in app.state.rag_tasks_flag:
                raise ValueError(f'Document {request.key_id} is running.')
            else:
                stop_event = Event()
                future = asyncio.create_task(async_task(task_id, stop_event, request))
                app.state.rag_tasks_flag[request.key_id] = (future, stop_event)
    except Exception as e:
        logger.error(f'{task_id} Fail to submit task to thread pool for:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to submit task to thread pool for:{type(e).__name__}: {str(e)}.'
        )
    
    logger.debug(f'{task_id} Submit task to thread pool.')

    # 返回响应
    return ApiRagResponseModel(
        code=1000,
        msg=f'{task_id} Success.',
    )