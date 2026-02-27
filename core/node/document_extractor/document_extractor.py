from typing import List, Union
from pathlib import Path
from uuid import uuid4

from core.rag.extractor.extract_processor import ExtractProcessor

async def extract_single_file(file_path: Path) -> str:
    file_id = str(uuid4())
    documents = ExtractProcessor.extract(file_path, file_id)
    return ''.join(document.page_content for document in documents)

async def aextract_document(path_prefix: str, file: Union[str, List[str]], **kwargs):
    output = {
        'data': {
            'text': None,  # Union[str, List[str]
        },
        'usage': None 
    }
    
    path_prefix = Path(path_prefix)

    if isinstance(file, str):
        file_path = path_prefix / file
        text = await extract_single_file(file_path)

    elif isinstance(file, List):
        texts = []
        files = file
        for file in files:
            file_path = path_prefix / file
            text = await extract_single_file(file_path)
            texts.append(text)
        text = texts
    else:
        raise ValueError(f"The parameter type is illegal.")
    
    output['data']['text'] = text

    return output