from datasets.tomography_paths import Dataset
import numpy as np
import mrcfile
from scipy.optimize import minimize_scalar


class TomographyDatasetSlices(Dataset):
    def __init__(self,
                 data_folder_path: list, 
                 max_files: int = None
                 ):
        super().__init__(data_folder_path,
                         max_files
                         )
        self.current_idx = 0
        self.height = None
        self.bacteria_name = None

    def __getitem__(self, _=None) -> dict:
        idx = self.current_idx
        self.tomogram = None
        with mrcfile.open(self.fpaths[idx]) as mrc:
            self.tomogram = mrc.data.copy()
        self.height = self.tomogram.shape[0]
        
        #Eliminate top and bottom of the tomogram
        margin = int(self.height * 0.2)
    
        self.tomogram = self.tomogram[margin:(self.height - margin), :, :]
        
        center = find_local_minima_near_center(self.tomogram)
        
        # Extract the middle mean slice
        middle_slice = self.tomogram[center[0]-5:center[0]+5, :, :].mean(0)
        
        # Get bacteria name from path
        self.bacteria_name = self.fpaths[idx].split("/")[-1].split(".")[0]
        
        self.current_idx += 1
        
        return {
            "key": self.bacteria_name,
            "data": [middle_slice, center]
        }
        

    def __len__(self):
        return len(self.fpaths) - self.current_idx
    
    def pop(self) -> dict:
        return self.__getitem__()

def find_local_minima_near_center(arr):
    def get_means(arr, axis):
        means = np.mean(arr, axis=axis)
        return means

    def fit_quadratic_function(means):
        # Fit a quadratic function: f(x) = ax^2 + bx + c
        x = np.arange(len(means))
        coeffs = np.polyfit(x, means, 2)
        return coeffs

    def quadratic_function(x, coeffs):
        a, b, c = coeffs
        return a * x**2 + b * x + c

    def find_local_minima(coeffs, center_index, length):
        # Define a lambda function for the quadratic function
        func = lambda x: quadratic_function(x, coeffs)
        # Use minimize_scalar to find the local minimum near the center slice
        # Bounds are set within the valid range of the means array
        result = minimize_scalar(func, bounds=(0, length-1), method='bounded')
        return result.x, result.fun
    
    means_x = get_means(arr, axis=0)
    means_y = get_means(arr, axis=1)
    means_z = get_means(arr, axis=2)

    center_x = arr.shape[0] // 2
    center_y = arr.shape[1] // 2
    center_z = arr.shape[2] // 2

    coeffs_x = fit_quadratic_function(means_x)
    coeffs_y = fit_quadratic_function(means_y)
    coeffs_z = fit_quadratic_function(means_z)

    print("Coefficients found")

    min_x, val_x = find_local_minima(coeffs_x, center_x, len(means_x))
    min_y, val_y = find_local_minima(coeffs_y, center_y, len(means_y))
    min_z, val_z = find_local_minima(coeffs_z, center_z, len(means_z))


    print("computed Local min")
    return [min_x, min_y, min_z]