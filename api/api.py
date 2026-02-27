import importlib
import os
import traceback
from contextlib import asynccontextmanager
from uuid import uuid4
from typing import Type, Callable, Dict, Tuple, AsyncGenerator, Coroutine
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import asyncio
import hashlib

from pydantic import BaseModel
from fastapi import FastAPI, Request, Body
from fastapi.responses import StreamingResponse

from core.model.model_manager import ModelManager, ModelInstanceType
from core.database.database_factory import DatabaseFactory
from api.base_model import ResponseModel
from config.config import get_config
from logger import get_logger


logger = get_logger('api')

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.type = 'inner'

        # 连接数据库
        db = None
        database_info = get_config()['dependent_info']['database']
        db = DatabaseFactory.get_database(database_info['type'])
        await db.connect(**database_info)
        logger.info('Succeed to connect to database.')

        # 初始化任务管理结构
        # rag任务队列
        # rag_tasks_flag:       Dict[int, Tuple[asyncio.Future, threading.Event]]
        app.state.rag_tasks_flag = {}
        app.state.rag_tasks_mtx = asyncio.Lock()
        # 智能体任务队列
        # agent_tasks_flag:     Dict[str, Tuple[asyncio.Future, threading.Event]]
        app.state.agent_tasks_flag = {}
        app.state.agent_tasks_mtx = asyncio.Lock()
        
        # 初始化线程池
        api_executor_threads = get_config()['server']['api_executor_threads']
        app.state.api_thread_pool = ThreadPoolExecutor(max_workers=api_executor_threads)
        logger.info(f'Succeed to init api thread pool with size({api_executor_threads}).')

        logger.info('Server started.')
        yield
    except Exception as e:
        logger.fatal(f'The initialization execution failed for:\n{traceback.format_exc()}')
    finally:
        logger.info('Server stopped.')

        # 关停Rag任务
        if hasattr(app.state, 'rag_tasks_mtx') and hasattr(app.state, 'rag_tasks_flag'):
            async with app.state.rag_tasks_mtx:
                logger.info(f'Prepare to stop rag tasks [{len(app.state.rag_tasks_flag)}]...')
                logger.info(f"Unfinished task_id: {list(app.state.rag_tasks_flag.keys())}")
                for idx, (future, stop_event) in app.state.rag_tasks_flag.items():
                    stop_event.set()
                    future.cancel()
                    logger.info(f'Stop rag task: {idx}.')
                logger.info('Notified rag tasks to stop.')
        
        # 关停Agent任务
        if hasattr(app.state, 'agent_tasks_mtx') and hasattr(app.state, 'agent_tasks_flag'):
            async with app.state.agent_tasks_mtx:
                logger.info(f'Prepare to stop agent tasks [{len(app.state.agent_tasks_flag)}]...')
                logger.info(f"Unfinished task_id: {list(app.state.agent_tasks_flag.keys())}")
                for idx, (future, stop_event) in app.state.agent_tasks_flag.items():
                    stop_event.set()
                    future.cancel()
                    logger.info(f'Stop agent task: {idx}.')
                logger.info('Notified agent tasks to stop.')

        # 关闭线程池
        if hasattr(app.state, 'api_thread_pool'):
            logger.info('Waiting api thread pool shutdown ...')
            app.state.api_thread_pool.shutdown(wait=True)
            logger.info('Succeed to shutdown api thread pool.')

        # 从数据库断开连接
        if db:
            try:
                await db.disconnect()
                logger.info('Succeed to disconnect from database.')
            except Exception:
                logger.warning(f'Fail to disconnect from database for:\n{traceback.format_exc()}')

app = FastAPI(lifespan=lifespan)
app.state.unhealth_event = threading.Event()

def get_api_client_tag(conn: Request) -> Tuple[str, str]:
    return f"{conn.client.host}:{conn.client.port}-{conn.scope.get('path', 'unknown')}", str(uuid4())

def get_request_md5(body: Dict) -> str:
    raw = json.dumps(body, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

def create_generator_handler(route: str, request_model: Type[BaseModel], func: Callable):
    async def handler(conn: Request, body: Dict = Body(...)):
        tag, task_id = get_api_client_tag(conn)
        logger.debug(f'{task_id} {tag}')

        # 格式校验
        try:
            logger.debug(f'{task_id} Request body:\n{json.dumps(body, indent=4, ensure_ascii=False)}')
            request = request_model.model_validate(body)
        except Exception as e:
            logger.error(f'{task_id} Fail to validate pydantic instance for:\n{traceback.format_exc()}')
            return ResponseModel(
                code=2000,
                msg=f'{task_id}: The request body field is incorrect.'
            )

        # 获取模型示例
        try:
            model_instance = ModelManager.get_model_instance(
                provider=request.model_instance_provider,
                model_type=ModelInstanceType.LLM,
                **request.model_instance_config
            )
            logger.debug(f'{task_id} Got model instance.')
        except Exception as e:
            logger.error(f'{task_id} Fail to get model instance for:\n{traceback.format_exc()}')
            return ResponseModel(
                code=2000,
                msg=f'{task_id}: {type(e).__name__}: {str(e)}'
            )

        if 'stream' in request.model_fields and request.stream:
            async def inner_streaming_generator(queue: asyncio.Queue):
                try:
                    # 直接调用func，得到异步生成器
                    async_gen = func(model_instance, **request.to_dict())

                    # 异步迭代
                    logger.debug(f'{task_id} Stream start.')
                    async for chunk in async_gen:
                        # logger.debug(f'{task_id} Chunk: {chunk}')
                        t = ResponseModel(code=1000, msg=f'{task_id}', data=chunk['data'], usage=chunk['usage'], done=chunk['done'])
                        await queue.put(f"data: {t.model_dump_json()}\n\n")

                except Exception as e:
                    logger.error(f'{task_id} Streaming error:\n{traceback.format_exc()}')
                    t = ResponseModel(code=2000, msg=f'{task_id} {str(e)}', done=True)
                    await queue.put(f"event: error\ndata: {t.model_dump_json()}\n\n")
                except asyncio.CancelledError:
                    logger.warning(f"{task_id} Task cancel.")
                finally:
                    await queue.put(None)
                    logger.debug(f'{task_id} Stream done.')

            async def streaming_generator():
                queue = asyncio.Queue()

                task = asyncio.create_task(
                    inner_streaming_generator(queue)
                )

                try:
                    while True:
                        if await conn.is_disconnected():
                            task.cancel()
                            break

                        item = await queue.get()
                        if item is None:
                            break

                        yield item

                except asyncio.CancelledError:
                    task.cancel()
                    raise
                finally:
                    logger.debug(f"{task_id} 'streaming_generator' done.")
                    task.cancel()

            return StreamingResponse(streaming_generator(), media_type="text/event-stream")
        else:
            # 调用
            try:
                logger.debug(f'{task_id} Processing.')
                # 此处ApiRequestModel.to_dict方法，只将自定义的字段转为dict，model相关的不转换
                async_r = func(model_instance, **request.to_dict())
                if isinstance(async_r, AsyncGenerator):
                    async for chunk in async_r:
                        result = chunk
                elif isinstance(async_r, Coroutine):
                    result = await async_r
                logger.debug(f'{task_id} Processed.')
            except Exception as e:
                logger.error(f'{task_id} Fail to call generate operation for:\n{traceback.format_exc()}')
                return ResponseModel(
                    code=2000,
                    msg=f'{task_id}: {type(e).__name__}: {str(e)}'
                )

            # 组合响应
            response = ResponseModel(
                code=1000,
                msg=f'{task_id}: Success.',
                data=result['data'],
                usage=result['usage'],
            )

            logger.debug(f'{task_id} Response:\n{response.model_dump_json(indent=4)}')
            return response

    app.post(route, response_model=ResponseModel, response_model_exclude_none=False)(handler)

def create_node_handler(route: str, request_model: Type[BaseModel], func: Callable):
    async def handler(conn: Request, body: Dict = Body(...)):
        tag, task_id = get_api_client_tag(conn)
        logger.debug(f'{task_id} {tag}')

        # 格式校验
        try:
            logger.debug(f'{task_id} Request body:\n{json.dumps(body, indent=4, ensure_ascii=False)}')
            request = request_model.model_validate(body)
        except Exception as e:
            logger.error(f'{task_id} Fail to validate pydantic instance for:\n{traceback.format_exc()}')
            return ResponseModel(
                code=2000,
                msg=f'{task_id}: The request body field is incorrect.'
            )

        if 'stream' in request.model_fields and request.stream:
            async def inner_streaming_generator(queue: asyncio.Queue):
                try:
                    async_gen = func(**request.to_dict())

                    # 异步迭代
                    logger.debug(f'{task_id} Stream start.')
                    async for chunk in async_gen:
                        # logger.debug(f'{task_id} Chunk: {chunk}')
                        t = ResponseModel(code=1000, msg=f'{task_id}', data=chunk['data'], usage=chunk['usage'], done=chunk['done'])
                        await queue.put(f"data: {t.model_dump_json()}\n\n")

                except Exception as e:
                    logger.error(f'{task_id} Streaming error:\n{traceback.format_exc()}')
                    t = ResponseModel(code=2000, msg=f'{task_id} {str(e)}', done=True)
                    await queue.put(f"event: error\ndata: {t.model_dump_json()}\n\n")
                except asyncio.CancelledError:
                    logger.warning(f"{task_id} Task cancel.")
                finally:
                    await queue.put(None)
                    logger.debug(f'{task_id} Stream done.')

            async def streaming_generator():
                queue = asyncio.Queue()

                task = asyncio.create_task(
                    inner_streaming_generator(queue)
                )

                try:
                    while True:
                        if await conn.is_disconnected():
                            task.cancel()
                            break

                        item = await queue.get()
                        if item is None:
                            break

                        yield item

                except asyncio.CancelledError:
                    task.cancel()
                    raise
                finally:
                    logger.debug(f"{task_id} 'streaming_generator' done.")
                    task.cancel()

            return StreamingResponse(streaming_generator(), media_type="text/event-stream")
        else:
            # 调用
            try:
                logger.debug(f'{task_id} Processing.')
                async_r = func(**request.to_dict())
                if isinstance(async_r, AsyncGenerator):
                    async for chunk in async_r:
                        result = chunk
                elif isinstance(async_r, Coroutine):
                    result = await async_r
                logger.debug(f'{task_id} Processed.')
            except Exception as e:
                logger.error(f'{task_id} Fail to call node operation for:\n{traceback.format_exc()}')
                return ResponseModel(
                    code=2000,
                    msg=f'{task_id}: {type(e).__name__}: {str(e)}'
                )

            # 组合响应
            response = ResponseModel(
                code=1000,
                msg=f'{task_id}: Success.',
                data=result['data'],
                usage=result['usage'],
            )

            logger.debug(f'{task_id} Response:\n{response.model_dump_json(indent=4)}')
            return response

    app.post(route, response_model=ResponseModel, response_model_exclude_none=False)(handler)


def auto_import_api():
    current_dir = os.path.dirname(__file__)
    package = __package__

    for filename in os.listdir(current_dir):
        if filename.startswith("api_") and filename.endswith(".py"):
            module_name = filename[:-3]
            full_module_name = f"{package}.{module_name}" if package else module_name
            try:
                importlib.import_module(full_module_name)
                logger.info(f"Imported API module: {full_module_name}")
            except Exception as e:
                logger.error(f"Failed to import {full_module_name}: {type(e).__name__}: {str(e)}")
                app.state.unhealth_event.set()

    for subdir in os.listdir(current_dir):
        subdir_path = os.path.join(current_dir, subdir)

        # 只处理以 api_ 开头的子目录
        if os.path.isdir(subdir_path) and subdir.startswith("api_"):
            for filename in os.listdir(subdir_path):
                if filename.startswith("api_") and filename.endswith(".py"):
                    module_name = filename[:-3]
                    full_module_name = f"{package}.{subdir}.{module_name}" if package else f"{subdir}.{module_name}"
                    try:
                        importlib.import_module(full_module_name)
                        logger.info(f"Imported API module: {full_module_name}")
                    except Exception as e:
                        logger.error(f"Failed to import {full_module_name}: {type(e).__name__}: {str(e)}")
                        app.state.unhealth_event.set()

auto_import_api()