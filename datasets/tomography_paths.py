from .dataset_base import Dataset
from typing import SupportsIndex
import subprocess
import random

class TomographyPaths(Dataset):
    def __init__(
            self,
            input_path,
            max_files
        ):
        super().__init__(
            input_path,
            max_files
        )

    def __len__(self):
        return len(self.fpaths)
    
    def pop(self, index: int = None):
        if index is None:
            # Pop the last key if no index is provided
            key = next(reversed(self.fpaths))
            return {"data": self.fpaths.pop(key)}
        else:
            # Pop the item with the specified key
            return {"data": self.fpaths.pop(index, None)}
        
    def __getitem__(self, index):
        return {
                "data": self.fpaths[index]
            }
      
    def get_sample_list(
        self, 
        samplesize: int
        ):
        """
        Get a random sample of the specified size from the list of files.
        
        Parameters:
        files_list (list): List of files to sample from.
        sample_size (int): The number of files to sample.
        
        Returns:
        list: A random sample of files.
        """
        # Get a list of keys from the dictionary
        keys = list(self.fpaths.keys())
        # Sample keys from the list
        sampled_keys = random.sample(keys, samplesize)
        # Retrieve the file paths corresponding to the sampled keys
        return [self.fpaths[key] for key in sampled_keys]
    

if __name__ == "__main__":
    
    l_paths = [#"/home/matiasgp/groups/grp_tomo_db1_d1/nobackup/archive/TomoDB1_d1/FlagellarMotor_P1", 
                    # "/home/matiasgp/groups/grp_tomo_db1_d2/nobackup/archive/TomoDB1_d2/FlagellarMotor_P2", 
                    "/home/matiasgp/groups/grp_tomo_db1_d3/nobackup/archive/TomoDB1_d3", 
                    "/home/matiasgp/groups/grp_tomo_db1_d4/nobackup/archive/TomoDB1_d4"
                ]
    
    paths = TomographyPaths(l_paths, 10)
    print(paths)
    print(paths.get_sample_list(10))
    