import os
import logging

try:
    # 使用支持多进程的日志轮转处理器
    from concurrent_log_handler import ConcurrentRotatingFileHandler as RotatingFileHandler
except ImportError:
    from logging.handlers import RotatingFileHandler



_LOG_PATH = os.path.join(os.path.dirname(__file__), './log')
_LOG_FILE = os.path.join(_LOG_PATH, 'server.log')
_LOG_LEVEL = logging.DEBUG
_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(thread)d - %(message)s'
_LOG_MAX_BYTES = 50 * 1024 * 1024
_LOG_BACKUP_COUNT = 10

os.makedirs(_LOG_PATH, exist_ok=True)


_logger_handler = RotatingFileHandler(_LOG_FILE, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT, encoding='utf-8')
_logger_handler.setLevel(_LOG_LEVEL)
_logger_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

_initialized_loggers = set()

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if name not in _initialized_loggers:
        logger.setLevel(_LOG_LEVEL)
        logger.addHandler(_logger_handler)
        _initialized_loggers.add(name)
    return logger