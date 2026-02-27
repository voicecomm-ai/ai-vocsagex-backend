from abc import ABC, abstractmethod
from typing import List, Callable
import copy
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.rag.entities.document import Document, ChildDocument
from core.rag.utils.rag_utils import escape_text, remove_leading_symbols

class BaseSplitter(ABC):

    @abstractmethod
    async def split_chunks(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError
    
    def _split_chunks(
        self, 
        documents: List[Document], 
        length_function: Callable[[str], int], 
        chunk_identifier: str, 
        chunk_size: int, 
        chunk_overlap: int = 0, 
        child: bool = False, 
        pic_insert_lf: bool = True,
        **kwargs
    ) -> List[Document]:
        '''
            使用langchain的简单通用切分
            Args: 
                child: 是否切分为子片段
                pic_insert_lf: 是否需要在图片前后插入换行
        '''
        # 提取page_content中的markdown图片链接
        markdown_image_pattern = r'(!\[.*?\]\(https?://[^\s)]+\))'
        image_placeholders: list[str] = []

        def replace_with_image_placeholder(match, image_placeholders=image_placeholders):
            url = match.group(1)
            placeholder = f"MarkdownImage{len(image_placeholders)}Url"
            image_placeholders.append(url)
            return f"{placeholder}"
        
        for document in documents:
            document.page_content = re.sub(markdown_image_pattern, replace_with_image_placeholder, document.page_content)

        # 提取page_content中的超链接
        markdown_hyperlink_pattern = r'\[(.*?)\]\((https?://[^\s)]+)\)'
        hyperlink_placeholders: list[str] = []

        def replace_with_hyperlink_placeholder(match, hyperlink_placeholders=hyperlink_placeholders):
            label = match.group(1)
            url = match.group(2)
            placeholder = f"MarkdownHyperlink{len(hyperlink_placeholders)}Url"
            hyperlink_placeholders.append(url)
            return f"[{label}]({placeholder})"
        
        for document in documents:
            document.page_content = re.sub(markdown_hyperlink_pattern, replace_with_hyperlink_placeholder, document.page_content)

        # 自定义切分
        new_documents = []
        if chunk_identifier:
            chunk_identifier = escape_text(chunk_identifier)
            for document in documents:
                page_content_list = document.page_content.split(chunk_identifier)
                for page_content in page_content_list:
                    new_document = copy.deepcopy(document)
                    new_document.page_content = page_content
                    new_documents.append(new_document)
        
        # langchain切分
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "。", ". ", " ", ""],
            keep_separator='end',
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )

        results = splitter.split_documents(new_documents)

        # 移除片段开头标点
        if not child:
            for result in results:
                result.page_content = remove_leading_symbols(result.page_content)

        if not child:
            # langchain_core.documents.Document -> core.rag.entities.document.Document
            final_documents = [
                Document(page_content=document.page_content, metadata=document.metadata) 
                for document in results
            ]
        else:
            # langchain_core.documents.Document -> core.rag.entities.document.ChildDocument
            final_documents = [
                ChildDocument(page_content=document.page_content, metadata=document.metadata) 
                for document in results
            ]

        # 回插超链接
        tag_hyperlink_pattern = r'MarkdownHyperlink(\d+)Url'
        def recovery_hyperlink(match):
            index = int(match.group(1))
            if 0 <= index < len(hyperlink_placeholders):
                return hyperlink_placeholders[index]
            return match.group(0)

        for final_document in final_documents:
            final_document.page_content = re.sub(tag_hyperlink_pattern, recovery_hyperlink, final_document.page_content)

        # 回插图片链接
        tag_image_pattern = r'MarkdownImage(\d+)Url'
        def recovery_image(match):
            index = int(match.group(1))
            if 0 <= index < len(image_placeholders):
                # 注意此处在图片的前后添加换行
                if pic_insert_lf:
                    return f'\n{image_placeholders[index]}\n'
                else:
                    return f'{image_placeholders[index]}'
            return match.group(0)

        for final_document in final_documents:
            final_document.page_content = re.sub(tag_image_pattern, recovery_image, final_document.page_content)
            # 移除图片前后的换行位于片段首尾的情况
            final_document.page_content = final_document.page_content.strip('\n')
            # 将多个\n替换为\n
            final_document.page_content = re.sub(r'\n{2,}', '\n', final_document.page_content)

        # 清理page_content为空字符串的document
        final_documents = [doc for doc in final_documents if doc.page_content]

        return final_documents
