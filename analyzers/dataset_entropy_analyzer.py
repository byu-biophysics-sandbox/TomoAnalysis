from analyzers import Analysis
import numpy as np                 # Import numpy for array operations
import os.path                     # Import os for operating system functions
from scipy.stats import entropy

class DatasetEntropyAnalyzer(Analysis):
    def __init__(self):
        pass
    
    def analyze(self, data, key, results: dict, **kwargs):
        try:
            data_pre = data
            data = min_max_normalize(data_pre) 
            data = histogram_equalization_3d(data)
            d_shp = data.shape  # Get shape of the numpy array
            
            # Extract the base name (file name with extension)
            base_name = os.path.basename(key)
            # Remove the .rec extension
            base_name = os.path.splitext(base_name)[0]
            
            entropy0, means0 = process_tomogram(data, 0, 2)
            entropy1, means1 = process_tomogram(data, 1, 2)
            entropy2, means2 = process_tomogram(data, 2, 2)
            
            
            seg_results = {
                "entropy_0": entropy0,
                "entropy_1": entropy1,
                "entropy_2": entropy2,
                "means_0": means0,
                "means_1": means1,
                "means_2": means2,
                "data": data,
                "data_pre": data_pre
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

def process_tomogram(tomogram, axis, slices_per_frame):
    """
    Process a 3D tomogram by calculating the entropy of slices along a specified axis,
    plotting the resulting images, and calculating the entropy values.

    Parameters:
    - tomogram (numpy.ndarray): The 3D tomogram data.
    - axis (int): The axis along which to process slices (0, 1, or 2).
    - slices_per_frame (int): The number of slices to process per frame.

    Returns:
    - entropy_values (list): A list of entropy values of the processed slices.
    """
    if axis not in [0, 1, 2]:
        raise ValueError("Axis must be 0, 1, or 2.")
    
    entropy_values = []
    num_slices = tomogram.shape[axis]

    for start in range(0, num_slices, slices_per_frame):
        end = min(start + slices_per_frame, num_slices)

        # Process the specified slices along the given axis
        if axis == 0:
            slice_data = tomogram[start:end, :, :]
        elif axis == 1:
            slice_data = tomogram[:, start:end, :]
        elif axis == 2:
            slice_data = tomogram[:, :, start:end]

        # Flatten the slice data
        flattened_slice = slice_data.flatten()
        
        # Calculate the entropy of the flattened slice
        slice_entropy = entropy(np.histogram(flattened_slice, bins=256)[0])
        entropy_values.append(slice_entropy)
    """
    Process a 3D tomogram by averaging slices along a specified axis,
    plotting the resulting images, and calculating the mean values.

    Parameters:
    - tomogram (numpy.ndarray): The 3D tomogram data.
    - axis (int): The axis along which to average slices (0, 1, or 2).
    - slices_per_frame (int): The number of slices to average per frame.

    Returns:
    - mean_values (list): A list of mean values of the averaged slices.
    """
    mean_values = []

    for start in range(0, num_slices, slices_per_frame):
        end = min(start + slices_per_frame, num_slices)

        # Average the specified slices along the given axis
        if axis == 0:
            averaged_slice = np.mean(tomogram[start:end, :, :], axis=0)
        elif axis == 1:
            averaged_slice = np.mean(tomogram[:, start:end, :], axis=1)
        elif axis == 2:
            averaged_slice = np.mean(tomogram[:, :, start:end], axis=2)

        
        # Calculate the mean value of the averaged slice
        mean_value = np.mean(averaged_slice)
        mean_values.append(mean_value)
    
    
    return entropy_values, mean_values
    



