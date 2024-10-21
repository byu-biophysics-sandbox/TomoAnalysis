from analyzers import Analysis
import numpy as np                 # Import numpy for array operations
import os.path                     # Import os for operating system functions
from writers import Slide, Video

class TomoStats(Analysis):
    def __init__(self):
        pass
    
    def analyze(self, data, key, results: dict, **kwargs):
        try:
            data = min_max_normalize(data) 
            data = histogram_equalization_3d(data)
            d_shp = data.shape  # Get shape of the numpy array
            min_val = np.min(data)  # Calculate minimum value in array
            max_val = np.max(data)  # Calculate maximum value in array
            mean = np.mean(data)  # Calculate mean value of array
            
            # Extract the base name (file name with extension)
            base_name = os.path.basename(key)
            # Remove the .rec extension
            base_name = os.path.splitext(base_name)[0]

        
            """Plot Video test"""
            test_plot = Video(data, str(base_name))
            test_plot.plot_video(20, d_shp[0], 15, 0)
            test_plot = Video(data, str(base_name))
            test_plot.plot_video(20, d_shp[1], 15, 1)
            test_plot = Video(data, str(base_name))
            test_plot.plot_video(20, d_shp[2], 15, 2)
            """End of test"""
            
            # """Plot Image Test"""
            # test_plot = Slide(data, str("/home/matiasgp/Desktop/Toma-Analysis/tests/"+base_name+".png"))
            # test_plot.plot_3d_img(1, int(d_shp[1]/2))
            # """End of test"""
            
            seg_results = {
                "shape": d_shp,
                "mean": mean,
                "min_val": min_val,
                "max_val": max_val,
                
            }
            results[key] = seg_results
                    
        except Exception as e:
            # Handle any exceptions that occur during processing
            print(f'Error processing {key}: {e}')
            
def min_max_normalize(array):
    # Convert to float32 to prevent overflow issues
    array = array.astype(np.float32)
    
    min_val = np.min(array)
    max_val = np.max(array)
    
    if min_val == max_val:
        # Avoid division by zero if the array contains a single unique value
        return np.zeros(array.shape, dtype=np.float32)
    
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

import numpy as np

def histogram_equalization_3d(image):
    """
    Apply histogram equalization to a 3D array.
    
    Args:
    - image (np.ndarray): 3D array representing the image or volume.
    
    Returns:
    - image_equalized (np.ndarray): Histogram-equalized 3D array.
    """
    # Flatten the 3D image array and calculate histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 1])

    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()  # Normalize CDF

    # Use linear interpolation of the CDF to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)

    # Reshape the flattened image back to the original 3D shape
    image_equalized = image_equalized.reshape(image.shape)
    
    return image_equalized

