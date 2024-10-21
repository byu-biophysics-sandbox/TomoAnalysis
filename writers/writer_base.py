from abc import ABC, abstractmethod

class Writer(ABC):
    def __init__(self, output_file="./myfile.csv"):
        self.file_path = output_file

    @abstractmethod
    def write(self, results: list, **kwargs):
        pass
    
    @abstractmethod
    def finish_and_save(self):
        pass

if __name__ == "__main__":
    writer = Writer()