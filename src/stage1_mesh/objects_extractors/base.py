from abc import ABC, abstractmethod
from typing import List

class BaseObjectsExtractor(ABC):
    @abstractmethod
    def extract(self, prompt: str) -> List[str]:
        """
        Input: "a red chair in front of the mirror"
        Output: ["red chair"]
        """
        pass