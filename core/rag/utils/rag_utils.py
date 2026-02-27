from typing import List, Dict
from uuid import uuid4
from hashlib import sha256
import re

from core.rag.entities.document import Document
from core.rag.splitter.splitter_entities import SplitType

def to_base36(num):
    chars = '0123456789abcdefghijklmnopqrstuvwxyz'
    if num == 0:
        return '0'
    res = ''
    while num > 0:
        num, i = divmod(num, 36)
        res = chars[i] + res
    return res

def short_unique_id(length=8):
    # 取uuid4的int部分，右移64位截断成64bit整数
    n = uuid4().int >> 64
    base36_str = to_base36(n)
    return base36_str[:length]

def escape_text(text: str) -> str:
    segments: list[tuple[str, bool]] = []
    for word in text:
        if segments and word.isascii() and segments[-1][1]:
            segments[-1] = (segments[-1][0] + word, True)
        else:
            segments.append((word, word.isascii()))
        
    output = ''
    for segment, is_ascii in segments:
        if is_ascii:
            try:
                output += segment.encode('utf-8').decode('unicode_escape')
            except:
                output += segment
        else:
            output += segment

    return output

def len_without_link(obj: str) -> int:
    markdown_image_pattern = r'!\[.*?\]\(https?://[^\s)]+\)'
    hyperlink_image_pattern = r'\[(.*?)\]\((https?://[^\s)]+)\)'
    return len(re.sub(hyperlink_image_pattern, r'\1', re.sub(markdown_image_pattern, '', obj)))

def remove_leading_symbols(text: str) -> str:
    pattern = r"^[\u2000-\u206F\u2E00-\u2E7F\u3000-\u303F!\"#$%&'()*+,./:;<=>?@^_`~]+"
    return re.sub(pattern, "", text)

def generate_chunks_from_documents(split_type: str, documents: List[Document])-> List[Dict]:
    return [document.to_dict() for document in documents]

def generate_preview_chunks_from_documents(split_type: str, documents: List[Document], is_preview: bool = False, is_edited: bool = False, chunk_status: str = 'ENABLE') -> List[Dict]:
    '''
        生成文档的预览片段
        NOTE:
        - 将`is_preview`总是设置为False
    '''
    if is_preview:
        preview_chunks = []
        
        if split_type == SplitType.NORMAL_QA.value:
            for i, document in enumerate(documents):
                item = {
                    'id': i + 1,
                    'question': document.page_content,
                    'answer': document.metadata.get('answer', ''),
                    'character': document.metadata['content_len'] + document.metadata['answer_len'],
                }
                preview_chunks.append(item)

        elif split_type == SplitType.ADVANCED_FULL_DOC.value or split_type == SplitType.ADVANCED_PARAGRAPH.value:
            for i, document in enumerate(documents):
                item = {
                    'id': i + 1,
                    'content': [],
                    'character': 0,
                }
                for child in document.children:
                    item['content'].append(child.page_content)
                    item['character'] += child.metadata['content_len']
                preview_chunks.append(item)

        else:
            for i, document in enumerate(documents):
                item = {
                    'id': i + 1,
                    'content': document.page_content,
                    'character': document.metadata['content_len'],
                }
                preview_chunks.append(item)
        return preview_chunks

    else:
        preview_chunks = []
        
        if split_type == SplitType.NORMAL_QA.value:
            for i, document in enumerate(documents):
                item = {
                    'id': document.metadata.get('idx', 99999),
                    'question': document.page_content,
                    'answer': document.metadata.get('answer', ''),
                    'character': document.metadata['content_len'] + document.metadata['answer_len'],
                    'isEdited': is_edited,
                    'status': chunk_status,
                }

                if 'primary_key' in document.metadata and document.metadata['primary_key']:
                    item['primary_key'] = document.metadata['primary_key']

                if 'failed_reason' in document.metadata and document.metadata['failed_reason']:
                    item['failed_reason'] = document.metadata['failed_reason']

                preview_chunks.append(item)

        elif split_type == SplitType.ADVANCED_FULL_DOC.value or split_type == SplitType.ADVANCED_PARAGRAPH.value:
            for i, document in enumerate(documents):
                item = {
                    'id': document.metadata.get('idx', 99999),
                    'content': [],
                    'character': 0,
                    'isEdited': is_edited,
                    'status': chunk_status,
                }
                for child_idx, child in enumerate(document.children):
                    child_item = {
                        'id': child.metadata.get('idx', 99999),
                        'content': child.page_content,
                        'character': child.metadata['content_len'],
                        'isEdited': is_edited,
                        'status': chunk_status,
                    }
                    if 'primary_key' in child.metadata and child.metadata['primary_key']:
                        child_item['primary_key'] = child.metadata['primary_key']

                    if 'failed_reason' in child.metadata and child.metadata['failed_reason']:
                        child_item['failed_reason'] = child.metadata['failed_reason']

                    item['content'].append(child_item)
                    item['character'] += child.metadata['content_len']
                preview_chunks.append(item)

        else:
            for i, document in enumerate(documents):
                item = {
                    'id': document.metadata.get('idx', 99999),
                    'content': document.page_content,
                    'character': document.metadata['content_len'],
                    'isEdited': is_edited,
                    'status': chunk_status,
                }
                if 'primary_key' in document.metadata and document.metadata['primary_key']:
                    item['primary_key'] = document.metadata['primary_key']

                if 'failed_reason' in document.metadata and document.metadata['failed_reason']:
                    item['failed_reason'] = document.metadata['failed_reason']
                    
                preview_chunks.append(item)
        return preview_chunks

def list_to_pgvector_str(vec: List[float]) -> str:
    return '[' + ','.join(f'{v}' for v in vec) + ']'

def get_text_id(text: str) -> str:
    return str(uuid4())

def get_text_hash(text: str) -> str:
    hash_text = str(text) + "Voicecomm"
    return sha256(hash_text.encode()).hexdigest()

def add_usage_dict(l: Dict, r: Dict) -> Dict:
    return {
        'prompt_tokens': l.get('prompt_tokens', 0) + r.get('prompt_tokens', 0),
        'completion_tokens': l.get('completion_tokens', 0) + r.get('completion_tokens', 0),
        'total_tokens': l.get('total_tokens', 0) + r.get('total_tokens', 0)
    }

def document_to_context(documents: List[Document]) -> List[str]:
    return [document.metadata.get('context_content', '') for document in documents]