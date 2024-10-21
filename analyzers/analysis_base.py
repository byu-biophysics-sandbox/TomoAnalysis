from abc import (
    ABC,
    abstractmethod
)

class Analysis(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def analyze(self, datapointer, results: list, **kwargs):
        pass
