from datasets.tomography_paths import Dataset
import numpy as np
import mrcfile

class TomographySlices(Dataset):
    def __init__(
            self,
            file_path: str
        ):
        self.tomogram = None
        with mrcfile.open(file_path) as mrc:
            self.tomogram = mrc.data.copy()

        self.height = self.tomogram.shape[0]
        margin = int(self.height * .2)
        self.bacteria_name = file_path.split("/")[-1].split(".")[0]
        self.tomogram = self.tomogram[margin:(self.height-margin), :, :]
        print(self.height)
        self.current_idx = 2

    def __getitem__(self, _=None) -> dict:
        idx = self.current_idx
        selected_slice = self.tomogram[idx-2:idx + 2, :, :].mean(0)
        self.current_idx +=1

        return {
            "key": idx,
            "data": selected_slice
        }

    def __len__(self):
        return self.height - self.current_idx - 2
    
    def pop(self) -> dict:
        return self.__getitem__()

