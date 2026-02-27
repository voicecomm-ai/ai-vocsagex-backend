from typing import Optional

from pydantic import BaseModel


class ChildDocument(BaseModel):

    page_content: str

    vector: Optional[list[float]] = None

    metadata: dict = {}

    def to_dict(self) -> dict:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
        }

class Document(BaseModel):

    page_content: str

    vector: Optional[list[float]] = None

    metadata: dict = {}

    children: Optional[list[ChildDocument]] = None

    def to_dict(self) -> dict:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children] if self.children else []
        }