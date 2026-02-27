from typing import (
    List, Dict, Tuple, Union, 
    Optional, Callable, Awaitable, Any, 
)
import traceback
import asyncio

from core.model.model_manager import ModelManager, ModelInstanceType
from core.memory.base_utils import (
    select_memory, 
    insert_memory, 
    delete_memory, 
    update_memory, 
)
from config.config import get_config
from logger import get_logger
from core.database.database_factory import DatabaseFactory
from core.generator.memory_extractor.memory_extractor import aextract_memory
from core.rag.utils.rag_utils import add_usage_dict
from core.generator.query_rewriter.query_rewriter import arewrite_query

logger = get_logger("memory")

class Memory:
    @classmethod
    async def   embedding(
        cls, 
        content: str, 
        model_instance_provider: str, 
        model_instance_config: Dict,
        model_parameters: Optional[Dict], 
        **kwargs
    ) -> Tuple[List, Dict]:
        if not content:
            raise ValueError("Empty content.")

        model_instance = ModelManager.get_model_instance(
            model_instance_provider, 
            ModelInstanceType.Embedding,
            **model_instance_config
        )

        result = await model_instance.ainvoke_text_embedding(
            texts=[content], 
            model_parameters=model_parameters, 
        )

        return result.embeddings[0], result.usage.to_dict()

    @classmethod
    async def extract(
        cls, 
        query: str,
        answer: str,
        existing_memory_list: List[str],
        model_instance_provider: str, 
        model_instance_config: Dict,
        model_parameters: Optional[Dict], 
        **kwargs
    ) -> Tuple[List[str], Dict]:
        if not query and not answer:
            return [], {}

        model_instance = ModelManager.get_model_instance(
            model_instance_provider, 
            ModelInstanceType.LLM,
            **model_instance_config
        )

        result = await aextract_memory(
            model_instance=model_instance, 
            model_parameters=model_parameters,
            query=query,
            answer=answer,
            existing_memory_list=existing_memory_list,
        )

        contents = [
            obj["content"]
            for obj in result["data"]["memory"]
            if obj["user_related"] and obj["content"]
        ]

        new_contents = []
        for content in contents:
            if content not in existing_memory_list:
                new_contents.append(content)

        logger.debug(f"Extract memory: {new_contents}")
        return new_contents, result["usage"]

    @classmethod
    async def filter(
        cls, 
        query: str, 
        memories: List[Dict], 
        model_instance_provider: str, 
        model_instance_config: Dict,
    ) -> Tuple[List[Dict], Dict]:
        raise NotImplementedError

    @classmethod
    async def insert(
        cls, 
        contents: List[str], 
        model_instance_provider: str, 
        model_instance_config: Dict,
        model_parameters: Optional[Dict], 
        application_id: int, 
        user_id: int, 
        agent_id: int, 
        data_type: str,
        **kwargs
    ) -> Tuple[List[int], Dict]:
        if not contents:
            return [], {}

        # 获取数据库实例
        database_info = get_config()["dependent_info"]["database"]
        db = DatabaseFactory.get_database(database_info["type"])

        sum_usage = {}
        vectors = []
        for content in contents:
            (vector, usage) = await cls.embedding(
                content, model_instance_provider, model_instance_config, model_parameters, 
            )
            sum_usage = add_usage_dict(sum_usage, usage)
            vectors.append(vector)

        ids = []
        for content, vector in zip(contents, vectors):
            id = await insert_memory(db, content, vector, application_id, user_id, agent_id, data_type)
            ids.append(id)

        return ids, sum_usage

    @classmethod
    async def delete(
        cls, 
        ids: List[int], 
        application_id: int, 
        user_id: int, 
        agent_id: int, 
    ) -> None:
        raise NotImplementedError
        
        if not ids:
            return
        
        # 获取数据库实例
        database_info = get_config()["dependent_info"]["database"]
        db = DatabaseFactory.get_database(database_info["type"])

        await delete_memory(
            db, 
            ids, 
            application_id, 
            user_id,
            agent_id,
        )

    @classmethod
    async def update(
        cls, 
        ids: List[int], 
        contents: List[str], 
        model_instance_provider: str, 
        model_instance_config: Dict, 
        application_id: int, 
        user_id: int, 
        agent_id: int, 
    ) -> Dict:
        raise NotImplementedError

        if not ids:
            return
        
        # 获取数据库实例
        database_info = get_config()["dependent_info"]["database"]
        db = DatabaseFactory.get_database(database_info["type"])

        sum_usage = {}
        vectors = []
        for content in contents:
            (vector, usage) = await cls.embedding(
                content, model_instance_provider, model_instance_config
            )
            sum_usage = add_usage_dict(sum_usage, usage)
            vectors.append(vector)

        for id, content, vector in zip(ids, contents, vectors):
            await update_memory(
                db=db, 
                id=id,
                content=content,
                vector=vector,
            )

        return sum_usage

    @classmethod
    async def retrieve(
        cls, 
        query: str, 
        chat_model_instance_provider: str, 
        chat_model_instance_config: Dict, 
        chat_model_parameters: Optional[Dict], 
        embedding_model_instance_provider: str, 
        embedding_model_instance_config: Dict, 
        embedding_model_parameters: Optional[Dict], 
        application_id: int, 
        user_id: int, 
        agent_id: int, 
        expired_time: Optional[str], 
        data_type: str,
        **kwargs
    ) -> Tuple[List[Dict], Dict]:
        # 获取数据库实例
        database_info = get_config()["dependent_info"]["database"]
        db = DatabaseFactory.get_database(database_info["type"])

        # 查询重写
        usage = {}
        try:
            model_instance = ModelManager.get_model_instance(
                chat_model_instance_provider, 
                ModelInstanceType.LLM,
                **chat_model_instance_config
            )

            result = await arewrite_query(
                model_instance=model_instance, 
                model_parameters=chat_model_parameters, 
                query=query
            )

            query = result["data"]["query"]
            usage = add_usage_dict(usage, result['usage'])
            logger.debug(f"Rewrite query: {query}")
        except Exception as e:
            logger.warning(f"Fail to rewrite query for:\n{traceback.format_exc()}")

        if query != "":
            # 生成向量
            (vector, embedding_usage) = await cls.embedding(
                query, 
                embedding_model_instance_provider,
                embedding_model_instance_config,
                embedding_model_parameters, 
            )
            usage = add_usage_dict(usage, embedding_usage)
            logger.debug("Generated embedding.")

            # 检索
            records = await select_memory(
                db,
                vector,
                application_id,
                user_id,
                agent_id,
                expired_time,
                data_type,
            )

            # 返回
            return records, usage
        else:
            return [], usage

    @classmethod
    def create_retrieve_afunc(
        cls, 
        chat_model_instance_provider: str, 
        chat_model_instance_config: Dict, 
        chat_model_parameters: Optional[Dict], 
        embedding_model_instance_provider: str, 
        embedding_model_instance_config: Dict, 
        embedding_model_parameters: Optional[Dict], 
        application_id: int, 
        user_id: int, 
        agent_id: int, 
        expired_time: Optional[str], 
        data_type: str,
        **kwargs
    ) -> Callable[[str], Awaitable[List[str]]]:
        task_id = kwargs.get("task_id")
        async def _arun(query: str) -> List[str]:
            try:
                logger.debug(f"{task_id} Retrieving memory...")
                (memories, usage) = await cls.retrieve(
                    query=query, 
                    chat_model_instance_provider=chat_model_instance_provider,
                    chat_model_instance_config=chat_model_instance_config, 
                    chat_model_parameters=chat_model_parameters,
                    embedding_model_instance_provider=embedding_model_instance_provider, 
                    embedding_model_instance_config=embedding_model_instance_config, 
                    embedding_model_parameters=embedding_model_parameters, 
                    application_id=application_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    expired_time=expired_time,
                    data_type=data_type,
                )

                outs = [
                    memory["content"]
                    for memory in memories
                    if memory["content"]
                ]
                logger.debug(f"{task_id} Retrieved memory: {outs}.")
            except Exception as e:
                logger.error(f"{task_id} Fail to retrieve memory for:\n{traceback.format_exc()}")
                return []
            except asyncio.CancelledError:
                raise

            return outs
        
        return _arun

    @classmethod
    def create_record_afunc(
        cls, 
        chat_model_instance_provider: str, 
        chat_model_instance_config: Dict, 
        chat_model_parameters: Optional[Dict], 
        embedding_model_instance_provider: str, 
        embedding_model_instance_config: Dict, 
        embedding_model_parameters: Optional[Dict], 
        application_id: int, 
        user_id: int, 
        agent_id: int, 
        data_type: str,
        **kwargs
    ) -> Callable[[str, List[str]], Awaitable[Any]]:
        task_id = kwargs.get("task_id")
        async def _arun(query: str, existing_memories: List[str]) -> Any:
            try:
                # 提取
                logger.debug(f"{task_id} Extracting memory...")
                (contents, usage) = await Memory.extract(
                    query=query,
                    answer="",
                    existing_memory_list=existing_memories,
                    model_instance_provider=chat_model_instance_provider,
                    model_instance_config=chat_model_instance_config,
                    model_parameters=chat_model_parameters, 
                )
                logger.debug(f"{task_id} Extracted memory:\n{contents}")

                # 插入
                if contents:
                    logger.debug(f"{task_id} Inserting memory...")
                    (ids, usage) = await Memory.insert(
                        contents=contents,
                        model_instance_provider=embedding_model_instance_provider,
                        model_instance_config=embedding_model_instance_config,
                        model_parameters=embedding_model_parameters, 
                        application_id=application_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        data_type=data_type,
                    )
                    logger.debug(f"{task_id} Memory primary keys:\n{ids}")
                else:
                    logger.debug(f"{task_id} Empty memory contents.")
            except Exception as e:
                logger.error(f"{task_id} Fail to record memory for:\n{traceback.format_exc()}")
            except asyncio.CancelledError:
                raise

        return _arun
