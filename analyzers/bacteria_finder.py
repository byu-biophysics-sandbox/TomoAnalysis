from analyzers import Analysis
import numpy as np                 # Import numpy for array operations
import os.path                     # Import os for operating system functions
from scipy.stats import entropy
from scipy.ndimage import center_of_mass, distance_transform_edt, label
import numpy as np
factor = 5

class bacteriaFinder(Analysis):
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
            
            rescaled_possitive_point, rescaled_negative = process_tomogram(data)
            
            
            seg_results = {
                "data": data,
                "data_pre": data_pre,
                "p_points": rescaled_possitive_point,
                "n_points": rescaled_negative,
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

def select_ranked_dark_group(array_3d, percentile=5, rank=1, connectivity=2):
    """
    Normalizes the 3D array, selects a group of dark values by size ranking, and modifies the array to retain
    only the values in the selected group, setting all others to 1.
    
    Parameters:
        array_3d (np.ndarray): The input 3D array.
        percentile (float): The percentile below which values are considered dark after normalization.
        rank (int): The rank of the group to select based on size (1 for largest, 2 for second largest, etc.).
        connectivity (int): The connectivity criterion (1 for direct neighbors, 2 for diagonal neighbors).

    Returns:
        modified_array (np.ndarray): The array with only the values in the selected group retained, others set to 1.
        z_index (int): The index of the shortest axis (used as the z-axis).
    """
    # Normalize the array to [0, 1]
    array_min = np.min(array_3d)
    array_max = np.max(array_3d)
    
    normalized_array = (array_3d - array_min) / (array_max - array_min) if array_max > array_min else np.zeros_like(array_3d)

    # Determine the shortest axis and get array dimensions
    dims = sorted(enumerate(array_3d.shape), key=lambda x: x[1])
    z_index, z_size = dims[0][0]  # Shortest dimension as z
    x_size, y_size = array_3d.shape[dims[1][0]], array_3d.shape[dims[2][0]]

    # Define the center region, keeping the z-axis intact
    x_trim_start, x_trim_end = int(0.1 * x_size), x_size - int(0.1 * x_size)
    y_trim_start, y_trim_end = int(0.1 * y_size), y_size - int(0.1 * y_size)

    if z_index == 0:
        restricted_region = (slice(None), slice(x_trim_start, x_trim_end), slice(y_trim_start, y_trim_end))
    elif z_index == 1:
        restricted_region = (slice(x_trim_start, x_trim_end), slice(None), slice(y_trim_start, y_trim_end))
    else:
        restricted_region = (slice(x_trim_start, x_trim_end), slice(y_trim_start, y_trim_end), slice(None))

    # Extract the central region and apply a dark mask based on the given percentile
    restricted_normalized_array = normalized_array[restricted_region]
    dark_mask = restricted_normalized_array < np.percentile(restricted_normalized_array, percentile)

    # Label connected regions
    structure = np.ones((3, 3, 3)) if connectivity == 2 else None
    labeled_array, num_features = label(dark_mask, structure=structure)

    if num_features == 0:
        # Return a default array if no dark groups are found
        return np.ones_like(array_3d), z_index

    # Rank the groups by size
    label_counts = np.bincount(labeled_array.flat)[1:]  # Exclude background
    ranked_groups = sorted(enumerate(label_counts, start=1), key=lambda x: x[1], reverse=True)

    if rank > len(ranked_groups):
        # Return an array of ones if the rank exceeds the number of groups
        return np.ones_like(array_3d), z_index

    # Get the mask of the ranked group in the restricted region
    ranked_group_label = ranked_groups[rank - 1][0]
    ranked_group_mask_restricted = labeled_array == ranked_group_label

    # Expand the restricted mask to the original array size
    ranked_group_mask = np.zeros_like(normalized_array, dtype=bool)
    ranked_group_mask[restricted_region] = ranked_group_mask_restricted

    # Modify the array: Retain values in the ranked group, set others to 1
    modified_array = np.where(ranked_group_mask, array_3d, 1)

    return modified_array, z_index
    
def max_entropy_slice(array, num_slices=10):
    """
    Find the slice with the maximum entropy in a 3D numpy array.
    
    Parameters:
    - array: 3D numpy array to analyze
    - num_slices: Number of slices to average for calculating the entropy

    Returns:
    - max_entropy_slice: The index of the slice with the maximum entropy
    """
    
    # Determine the shortest axis
    shortest_axis = np.argmin(array.shape)
    
    # Variables to track the slice with the largest entropy value
    max_entropy_value = -np.inf
    max_entropy_slice = -1

    # Iterate over the slices along the shortest axis
    for frame in range(array.shape[shortest_axis]):
        
        # Depending on the axis, extract and average the slices
        if shortest_axis == 0:
            start_slice = max(0, frame - num_slices // 2)
            end_slice = min(array.shape[shortest_axis], frame + num_slices // 2 + 1)
            slices = array[start_slice:end_slice, :, :]
            slice_ = np.mean(slices, axis=0)
        elif shortest_axis == 1:
            start_slice = max(0, frame - num_slices // 2)
            end_slice = min(array.shape[shortest_axis], frame + num_slices // 2 + 1)
            slices = array[:, start_slice:end_slice, :]
            slice_ = np.mean(slices, axis=1)
        elif shortest_axis == 2:
            start_slice = max(0, frame - num_slices // 2)
            end_slice = min(array.shape[shortest_axis], frame + num_slices // 2 + 1)
            slices = array[:, :, start_slice:end_slice]
            slice_ = np.mean(slices, axis=2)
        
        # Flatten the slice and calculate entropy
        flattened_slice = slice_.flatten()
        hist = np.histogram(flattened_slice, bins=256)[0]
        entropy_value = entropy(hist, base=2)  # Calculate entropy using base-2
        
        # Update if this slice has the largest entropy value so far
        if entropy_value > max_entropy_value:
            max_entropy_value = entropy_value
            max_entropy_slice = frame

    # Return the index of the slice with the maximum entropy
    return max_entropy_slice


def rescale_point(original_dims, scaled_dims, scaled_point):
    """
    Rescales a point from a downscaled 3D matrix back to its original dimensions, 
    where the axes are defined as 0=z, 1=x, and 2=y.
    
    Parameters:
    - original_dims: Tuple of the original matrix dimensions (Z_orig, X_orig, Y_orig).
    - scaled_dims: Tuple of the downscaled matrix dimensions (Z_scaled, X_scaled, Y_scaled).
    - scaled_point: Tuple of the point coordinates in the downscaled matrix (z', x', y').
    
    Returns:
    - Tuple of the rescaled point coordinates (z, x, y).
    """
    # Extract the original and scaled dimensions
    Z_orig, X_orig, Y_orig = original_dims
    Z_scaled, X_scaled, Y_scaled = scaled_dims
    
    # Extract the downscaled point coordinates
    z_prime, x_prime, y_prime = scaled_point
    
    # Calculate the scaling factors for each dimension
    S_z = Z_orig / Z_scaled
    S_x = X_orig / X_scaled
    S_y = Y_orig / Y_scaled
    print(S_z, S_x, S_y)
    # Rescale the point
    z = z_prime * S_z
    x = x_prime * S_x
    y = y_prime * S_y
    
    return z, x, y

def organize_scaled_point(centroid, entropy_slice, s_axis):
    scaled_point = []
    if s_axis == 0:
        scaled_point = [entropy_slice, centroid[0], centroid[1]]
    elif s_axis == 1:
        scaled_point = [centroid[0], entropy_slice, centroid[1]]
    elif s_axis == 2:
        scaled_point = [centroid[0], centroid[1], entropy_slice]
    
    return scaled_point
    
def process_tomogram(o_tomo):
    
    # 1. Downsample tomogram, get the highest ranked dark connecting group.

    tomo = downsample_3d_average(o_tomo, 5)
    
    mask, s_axis = select_ranked_dark_group(tomo, percentile=10)

    # 2. Get highest entropy slice from the mask, trim the tomogram to look at the 40% part of the tomogram on each direction of the center slice.
    
    entropy_slice =  max_entropy_slice(mask, num_slices=10)

    # 3. Get centroids and re-scale

    positive_points, negative_points = find_points(mask, slice_number=entropy_slice)
    
    scaled_point = organize_scaled_point(entropy_slice, positive_points, s_axis)

    rescaled_possitive_point = rescale_point(o_tomo.shape, mask.shape, scaled_point)
    
    rescaled_negative = [factor * np.array(n) for n in negative_points]

    # 5.Return center possitive and negative coordinates
    

    return rescaled_possitive_point, rescaled_negative

        

    



