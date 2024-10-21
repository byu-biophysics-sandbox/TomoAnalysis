from analyzers import Analysis
from writers import Writer
from utils.thread_manager import ThreadManager
from utils.checkpointer import Checkpointer
from datasets import Dataset
import tqdm

class DataFactory:
    def __init__(
        self,
        analyzer: Analysis,
        writer: Writer,
        batch_size: int = 100,
        checkpointer: Checkpointer = None
    ):
        self.__threader = ThreadManager(analyzer)
        self.__writer = writer
        self.__batch_size = batch_size
        self.__checkpointer = checkpointer
    
    def process(
        self,
        dataset: Dataset,
        loading_bar_string: str
    ):
        batch_num = 1
        loading_bar = tqdm.tqdm(total=len(dataset), desc=loading_bar_string)

        while len(dataset) != 0:
            # Load the current batch
            current_batch = []
            while len(current_batch) < self.__batch_size and len(dataset) != 0:
                current_batch.append(dataset.pop()) # this should append a list of dictionaries
            
            # Submit batch
            batch_results = self.__threader.submit_batch(current_batch)
            
            # Write the results to the file.
            #self.__writer.write(batch_results)

            loading_bar.update(len(batch_results))

            if self.__checkpointer:
                if not self.__checkpointer.interval % batch_num:
                    self.__checkpointer.checkpoint()
                
            batch_num += 1
            
            return batch_results
            
        self.__writer.finish_and_save()
        
        

