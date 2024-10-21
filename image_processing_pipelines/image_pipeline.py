from abc import ABC, abstractmethod
class ImagePipeline(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def process_image(
        self,
        image,
        **kwargs
    ):
        pass