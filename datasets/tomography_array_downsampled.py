from datasets.tomography_paths import Dataset
import numpy as np
import mrcfile
import random
import sys

class TomographyArrayDownsampled(Dataset):
    def __init__(self, data_folder_path: list, max_files: int = None, factor = 5):
        super().__init__(data_folder_path, max_files)
        self.current_idx = 0
        self.height = None
        self.bacteria_name = None
        self.factor = factor

        # Ensure fpaths is initialized
        if not hasattr(self, 'fpaths'):
            self.fpaths = []

        # Debugging: Print number of files found
        print(f"Initialized with {len(self.fpaths)} files.")

    def __getitem__(self, idx):
        if idx >= len(self.fpaths) or idx < 0:
            raise IndexError("Index out of range")

        with mrcfile.open(self.fpaths[idx]) as mrc:
            tomogram = mrc.data.copy()
            downsampled_tomogram = downsample_3d_average(tomogram, self.factor)

        bacteria_name = self.fpaths[idx].split("/")[-1].split(".")[0]

        return {
            "key": self.fpaths[idx],
            "data": downsampled_tomogram,
            "bacteria_name": bacteria_name
        }

    def __len__(self):
        return len(self.fpaths)

    def pop(self):
        if self.current_idx >= len(self.fpaths):
            print("NO MORE FILES TO ANALYZE...")
            sys.exit()

        result = self.__getitem__(self.current_idx)
        self.current_idx += 1
        return result

    def random_pop(self):
        if len(self.fpaths) == 0:
            raise IndexError("No files available for random_pop()")

        idx = random.randint(0, len(self.fpaths) - 1)
        return self.__getitem__(idx)

def downsample_3d_average(image_3d, factor):
    """
    Downsample a 3D image by averaging non-overlapping blocks using vectorized operations.
    
    Parameters:
    - image_3d: 3D numpy array to downsample
    - factor: Downsampling factor (e.g., factor=8 means reducing the size by 1/8th)
    
    Returns:
    - Downsampled 3D numpy array
    """
    # Calculate the new shape considering integer division
    new_shape = (
        image_3d.shape[0] // factor,
        image_3d.shape[1] // factor,
        image_3d.shape[2] // factor
    )
    
    # Reshape the array by splitting into blocks
    reshaped = image_3d[:new_shape[0] * factor, 
                        :new_shape[1] * factor, 
                        :new_shape[2] * factor].reshape(
        new_shape[0], factor,
        new_shape[1], factor,
        new_shape[2], factor
    )
    
    # Average along the factor dimensions
    downsampled = reshaped.mean(axis=(1, 3, 5))
    
    return downsampled