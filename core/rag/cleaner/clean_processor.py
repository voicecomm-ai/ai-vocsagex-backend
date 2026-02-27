from typing import Optional

from core.rag.cleaner.cleaner_base import BaseCleaner
from core.rag.cleaner.cleaner_normal import NormalCleaner

class CleanProcessor:

    @classmethod
    def clean(cls, text: str, **kwargs) -> str:

        cleaner: Optional[BaseCleaner] = None

        cleaner = NormalCleaner(**kwargs)

        return cleaner.clean(text)