from enum import StrEnum

class SplitType(StrEnum):
    NORMAL = "NORMAL"
    '''普通分段'''
    NORMAL_QA = "NORMAL_QA"
    '''普通分段，Q&A'''
    ADVANCED_FULL_DOC = "ADVANCED_FULL_DOC"
    '''高级分段，父段全文'''
    ADVANCED_PARAGRAPH = "ADVANCED_PARAGRAPH"
    '''高级分段，父段段落'''