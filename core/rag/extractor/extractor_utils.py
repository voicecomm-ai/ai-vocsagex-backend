import concurrent.futures
from pathlib import Path
from typing import NamedTuple, Optional, cast
import os

class FileEncoding(NamedTuple):
    """A file encoding as the NamedTuple."""

    encoding: Optional[str]
    """The encoding of the file."""
    confidence: float
    """The confidence of the encoding."""
    language: Optional[str]
    """The language of the file."""


def detect_file_encodings(file_path: str, timeout: int = 5) -> list[FileEncoding]:
    """Try to detect the file encoding.

    Returns a list of `FileEncoding` tuples with the detected encodings ordered
    by confidence.

    Args:
        file_path: The path to the file to detect the encoding for.
        timeout: The timeout in seconds for the encoding detection.
    """
    import chardet

    def read_and_detect(file_path: str) -> list[dict]:
        rawdata = Path(file_path).read_bytes()
        return cast(list[dict], chardet.detect_all(rawdata))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(read_and_detect, file_path)
        try:
            encodings = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Timeout reached while detecting encoding for {file_path}")

    if all(encoding["encoding"] is None for encoding in encodings):
        raise RuntimeError(f"Could not detect encoding for {file_path}")
    return [FileEncoding(**enc) for enc in encodings if enc["encoding"] is not None]

_CONTENT_MAX_LENGTH = 15 * 1024 * 1024

def get_content_from_url(url: str, retries: int = 3, backoff: int = 2, timeout: int = 5, timeout_connect: int = 5, timeout_read: int = 5, timeout_write: int = 5, **kwargs):
    import httpx
    from httpx import Response

    timeout_config = httpx.Timeout(
        timeout=timeout,
        connect=timeout_connect,
        read=timeout_read,
        write=timeout_write
    )

    if 'follow_redirects' not in kwargs:
        kwargs['follow_redirects'] = True
    if 'verify' not in kwargs:
        kwargs['verify'] = True
    kwargs['timeout'] = timeout_config

    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=kwargs['timeout'], verify=kwargs['verify'], follow_redirects=kwargs['follow_redirects']) as client:

                # response = client.get(url)
                # response.raise_for_status()
                # return response
            
                with client.stream('GET', url) as response:
                    response.raise_for_status()
                    content = bytearray()
                    for chunk in response.iter_bytes():
                        content += chunk
                        if len(content) > _CONTENT_MAX_LENGTH:
                            raise ValueError(f"Content size exceeded {_CONTENT_MAX_LENGTH} bytes.")
                    return Response(
                        status_code=response.status_code,
                        headers=response.headers,
                        content=bytes(content)
                    )
        except httpx.RequestError as e:
            # print(f"[重试 {attempt+1}/{retries}] 请求错误: {e}")
            continue
        except httpx.HTTPStatusError as e:
            # print(f"[重试 {attempt+1}/{retries}] 状态码错误: {e.response.status_code}")
            continue
        except Exception as e:
            # print(f"[重试 {attempt+1}/{retries}] 未知错误: {e}")
            continue
    raise RuntimeError(f"Reached maximum retries ({retries}) for URL {url}")

def save_image(path: str, content):
    
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(content)