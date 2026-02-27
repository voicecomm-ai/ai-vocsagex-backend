from typing import Type, Dict
import threading

from core.database.database_base import BaseDatabase
from core.database.database_postgres import PostgresDatabase

_database_map: Dict[str, Type[BaseDatabase]] = {
    'postgres': PostgresDatabase
}

class DatabaseFactory:
    _instances = {}
    _lock = threading.Lock()

    @staticmethod
    def get_database(db_type: str, **kwargs) -> BaseDatabase:
        if db_type not in _database_map.keys():
            raise ValueError(f'Unsupported database type: {db_type}')

        if db_type not in DatabaseFactory._instances:
            with DatabaseFactory._lock:
                if db_type not in DatabaseFactory._instances:
                    DatabaseFactory._instances[db_type] = _database_map[db_type](**kwargs)
        return DatabaseFactory._instances[db_type]
