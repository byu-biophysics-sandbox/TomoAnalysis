import os
from abc import ABC, abstractmethod
import subprocess

class Dataset(ABC):
    def __init__(self, data_folder_path: list, max_files: int = None):
        self.max_files = max_files if max_files is not None else float('inf')
        self.fpaths = []  # List to store clean results
        
        file_count = 0  # To keep track of the number of files processed
        
        for path in data_folder_path:
            try:
                # Run the lfs find command to locate .rec files
                result = subprocess.run(["lfs", "find", path, "-type", "f", "--name", "*.rec"],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        check=True) 
                # Process the output of the command
                for name in result.stdout.splitlines():
                    if file_count >= self.max_files:
                        break

                    components = name.split('/')
                    found_peet = False
                    for component in components:
                        if 'peet' in component.lower() or "align" in component.lower():
                            found_peet = True
                            break
                    if not found_peet:
                        self.fpaths.append(name)
                        file_count += 1

            except subprocess.CalledProcessError as e:
                print(f"An error occurred while running the subprocess: {e}")
    
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def pop(self, index: int = -1):
        pass