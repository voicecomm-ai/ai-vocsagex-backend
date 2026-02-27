import json
from typing import Any, Mapping, Optional
import threading

from frozendict import frozendict

from logger import get_logger

logger = get_logger('config')

class Config:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, path: str):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_config(path)
        return cls._instance

    def _init_config(self, path: str):
        with open(path, encoding='utf-8') as f:
            self._config = frozendict(json.load(f))

    @property
    def config(self) -> Mapping[str, Any]:
        return self._config


# 模块级缓存变量
_config_instance: Optional[Config] = None

def init_config(path: str):
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(path)

def get_config() -> Mapping[str, Any]:
    if _config_instance is None:
        logger.fatal('Config not initialized. Call init_config(path) first.')
        raise RuntimeError("Config not initialized. Call init_config(path) first.")
    return _config_instance.config
