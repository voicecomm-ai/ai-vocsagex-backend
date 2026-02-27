from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    
    @abstractmethod
    def extract(self):
        raise NotImplementedError