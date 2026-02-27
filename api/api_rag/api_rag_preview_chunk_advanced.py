'''
    分段预览：高级分段
'''

from typing import Dict, Optional
import traceback
from uuid import uuid4
import json
from pathlib import Path

from pydantic import Field, BaseModel
from fastapi import Request, Body

from api.api import app, get_api_client_tag
from core.rag.extractor.extract_processor import ExtractProcessor
from core.rag.cleaner.clean_processor import CleanProcessor
from core.rag.splitter.split_processor import SplitProcessor
from core.rag.utils.rag_utils import generate_preview_chunks_from_documents
from logger import get_logger
from config.config import get_config

logger = get_logger('api')

class ApiRagRequestModel(BaseModel):
    class FatherChunkSetting(BaseModel):
        fulltext: bool = Field(..., description='父段是否采用全文')
        chunk_identifier: Optional[str] = Field('\n\n', description='分段标识符，父段采用全文时不填')
        chunk_size: Optional[int] = Field(500, description='分段最大长度，父段采用全文时不填')

    class SonChunkSetting(BaseModel):
        chunk_identifier: str = Field(..., description='分段标识符')
        chunk_size: int = Field(..., description='分段最大长度')

    class CleanerSetting(BaseModel):
        filter_blank: bool = Field(..., description='是否过滤空格符、换行符、制表符')
        remove_url: bool = Field(..., description='是否删除所有URL和电子邮件地址')

    key_id: int = Field(..., description='知识库所在表的主键，对应java的文档id')
    file_url: str = Field(..., description='文件url')
    fatherchunk_setting: FatherChunkSetting = Field(..., description='父段设置')
    sonchunk_setting: SonChunkSetting = Field(..., description='子段设置')
    cleaner_setting: CleanerSetting = Field(..., description='文本预处理规则')

class ApiRagResponseModel(BaseModel):
    code: int = Field(..., description='状态码，成功为1000，失败为2000')
    msg: str = Field(..., description='状态信息')
    data: Optional[Dict] = Field(None, description='数据')
    usage: Optional[Dict] = Field(None, description='模型tokens用量')

@app.post(path='/Voicecomm/VoiceSageX/Rag/PreviewChunkAdvanced', response_model=ApiRagResponseModel, response_model_exclude_none=False)
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

    # 提取
    file_id = str(uuid4())
    try:
        document_path_prefix = get_config().get('dependent_info').get('knowledge_base').get('document_path_prefix')
        file_path = Path(document_path_prefix) / request.file_url
        documents = ExtractProcessor().extract(file_path=file_path, file_id=file_id)
    except Exception as e:
        logger.error(f'{task_id} Fail to extract file:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to extract file for:{type(e).__name__}: {str(e)}.'
        )

    # 清洗
    try:
        for document in documents:
            document.page_content = CleanProcessor.clean(document.page_content, **(request.cleaner_setting.model_dump()))
    except Exception as e:
        logger.error(f'{task_id} Fail to clean file:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to clean file for:{type(e).__name__}: {str(e)}.'
        )

    # 分段
    try:
        split_type, documents = await SplitProcessor.split(documents, **request.model_dump())
    except Exception as e:
        logger.error(f'{task_id} Fail to split file:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to split file for:{type(e).__name__}: {str(e)}.'
        )
    
    # 为document生成序号
    for f_idx, f_document in enumerate(documents):
        f_document.metadata['idx'] = f_idx + 1
        if f_document.children:
            for s_idx, s_document in enumerate(f_document.children):
                s_document.metadata['idx'] = s_idx + 1
                s_document.metadata['f_idx'] = f_document.metadata['idx']

    try:
        # 组建响应
        preview_chunks = generate_preview_chunks_from_documents(split_type, documents)
        # logger.debug(f'{task_id} Preview chunks: {preview_chunks}')

        data = {
            'key_id': request.key_id,
            'file_url': request.file_url,
            'file_id': file_id,
            'chunk_type': split_type,
            'chunk': preview_chunks,
        }

    except Exception as e:
        logger.error(f'{task_id} Fail to generate response:\n{traceback.format_exc()}')
        return ApiRagResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to generate response for:{type(e).__name__}: {str(e)}.'
        )
    
    logger.debug(f'{task_id} Done.')

    return ApiRagResponseModel(
        code=1000,
        msg=f'{task_id}: Success.',
        data=data,
        usage=None,
    )