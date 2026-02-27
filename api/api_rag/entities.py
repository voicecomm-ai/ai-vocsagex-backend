from enum import Enum

class DocumentStatus(Enum):
    WAITING = 'WAITING'
    IN_PROGRESS = 'IN_PROGRESS'
    SUCCESS = 'SUCCESS'
    FAILED = 'FAILED'

class ChunkStatus(Enum):
    WAITING = 'WAITING'
    IN_PROGRESS = 'IN_PROGRESS'
    SUCCESS = 'SUCCESS'
    FAILED = 'FAILED'