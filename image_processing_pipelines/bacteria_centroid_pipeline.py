from image_processing_pipelines.image_pipeline import ImagePipeline
from utils.image_processor import ImageProcessor
import cv2
import os
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


class BacteriaCentroidPipeline(ImagePipeline):
    def __init__(
            self,
            downsample_factor:int=5,
            dark_group_percentile:int=10,
            entropy_slices_to_average:int=10,
            num_negative_points:int=3,
            num_positive_points:int=2,
            temp_image_database:str="temp_image_database",
        ):
        self.downsample_factor = downsample_factor
        self.dark_group_percentile = dark_group_percentile
        self.entropy_slices_to_average = entropy_slices_to_average
        self.num_positive_points = num_positive_points - 1
        self.num_negative_points = num_negative_points
        self.temp_image_database = temp_image_database
    
    def process_tomo(
            self,
            o_tomo,
            key,
            **kwargs
        ):
        
        temp_image_database = temp_image_database + str(key)
        
        # 1. Downsample tomogram, get the highest ranked dark connecting group.
        
        tomo = downsample_3d_average(o_tomo, downsample_factor)
        
        binary_mask, mask, s_axis = select_ranked_dark_group(tomo, percentile=dark_group_percentile)

        # 2. Get highest entropy slice from the mask, trim the tomogram to look at the 40% part of the tomogram on each direction of the center slice.
        entropy_slice =  max_entropy_slice(mask, num_slices=entropy_slices_to_average)

        # 3. Get positive and negative points, plot
        
        positive_points, negative_points = find_points(mask, slice_number=entropy_slice, num_negative_points=num_negative_points, num_positive_points=num_positive_points, min_distance_percent=0.1)

        # 4. Use points for SAM --> re-scale point to numpy
        
        scaled_entropy_slice = factor*entropy_slice
        
        np_possitive_points = upscale_points(factor, positive_points)
        
        np_negative_points = upscale_points(factor, negative_points)
        
        #5. Generate JPEG images for analysis, get size
        
        jpeg_shape = generate_images(o_tomo, save_path=temp_image_database)
        
        # Assuming you are slicing along the first axis (depth), use the height and width
        
        original_height, original_width = o_tomo.shape[1], o_tomo.shape[2]

        print("Frame:", scaled_entropy_slice)

        #6. Re-scale points for images
        jpeg_possitive_points = rescale_coordinates((original_height, original_width), jpeg_shape, np_positive_points)
        print("JPEG Positive:", jpeg_possitive_points)

        jpeg_negative_points = rescale_coordinates((original_height, original_width), jpeg_shape, np_negative_points)
        print("JPEG Negative:", jpeg_negative_points)


        #7. Organize feeding points to return
        
        jpeg_feeding_points = {
            scaled_entropy_slice: {
                "obj_id": 1,
                "points": np.concatenate((jpeg_possitive_points, jpeg_negative_points), axis=0).astype(np.float32),
                "labels": np.array([1] * len(jpeg_possitive_points) + [0] * len(jpeg_negative_points), dtype=np.int32)
            },
        }
        
        np_feeding_points = {
            scaled_entropy_slice: {
                "obj_id": 1,
                "points": np.concatenate((np_possitive_points, np_negative_points), axis=0).astype(np.float32),
                "labels": np.array([1] * len(np_possitive_points) + [0] * len(np_negative_points), dtype=np.int32)
            },
        }

        #8. Get entropy values across all slices, use k-means clustering to determine if values are too low. 
        # Determine entropy_index_stopper_values
        
        entropy_values = entropy_of_slices(mask)
        
        entropy_limits = slices_below_clustering(entropy_values)
        
        stopper_values = get_sequence_boundaries(entropy_limits)
        
        
        return {
            "jpeg_feeding_points": jpeg_feeding_points,
            "np_feeding_points": np_feeding_points,
            "stopper_points": stopper_vales,
            "temp_bacteria_path": temp_image_database,
        }
    
    def downsample_3d_average(
        image_3d, 
        factor
        ):
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
    
    def select_ranked_dark_group(
        array_3d,
        percentile=5,
        rank=1,
        connectivity=2
        ):
        """
        Normalizes the 3D array, then selects a specific group of dark values based on size ranking.
        Finds the darkest group only within the middle 60% of the x and y dimensions after trimming 20% from each side,
        while keeping the z-axis intact. Returns a mask of the ranked dark group, modifies the array to keep only the values
        in the ranked group while setting all other values to 1. Additionally, returns the size of the ranked group and the normalized array.

        Parameters:
            array_3d (np.ndarray): The input 3D array.
            percentile (float): The percentile below which values are considered dark, after normalization.
            rank (int): The rank of the group to return based on size (1 for largest, 2 for second largest, etc.).
            connectivity (int): The connectivity criterion (1 for direct neighbors, 2 for diagonal neighbors).

        Returns:
            ranked_group_mask (np.ndarray): A binary mask where the ranked dark group is marked as 1.
            ranked_group_size (int): The size of the ranked dark group.
            normalized_array (np.ndarray): The normalized array with values in the range [0, 1].
            modified_array (np.ndarray): The array with values outside the selected group set to 1, and values in the group retained.
        """
        # Normalize the array to the range [0, 1]
        array_min = np.min(array_3d)
        array_max = np.max(array_3d)
        
        # Prevent division by zero in case all values are the same
        if array_max - array_min != 0:
            normalized_array = (array_3d - array_min) / (array_max - array_min)
        else:
            normalized_array = np.zeros_like(array_3d)

        # Get the shape of the array and dynamically set z_size to be the shortest axis
        dims = sorted(enumerate(array_3d.shape), key=lambda x: x[1])  # Sort dimensions by size
        z_index, z_size = dims[0]  # The shortest dimension will be assigned to z_size
        other_indices = [dims[1][0], dims[2][0]]  # The other two dimensions

        # Assign the other dimensions to x_size and y_size based on their original positions
        x_size = array_3d.shape[other_indices[0]]
        y_size = array_3d.shape[other_indices[1]]

        # Calculate the limits for the centered 60% region along the x and y axes
        x_trim_start = int(0.1 * x_size)
        x_trim_end = x_size - x_trim_start
        
        y_trim_start = int(0.1 * y_size)
        y_trim_end = y_size - y_trim_start

        # Create the restricted region based on which axis is the shortest
        if z_index == 0:  # If the shortest dimension is x (original x-axis)
            restricted_region = (slice(None), slice(x_trim_start, x_trim_end), slice(y_trim_start, y_trim_end))
        elif z_index == 1:  # If the shortest dimension is y (original y-axis)
            restricted_region = (slice(x_trim_start, x_trim_end), slice(None), slice(y_trim_start, y_trim_end))
        elif z_index == 2:  # If the shortest dimension is z (original z-axis)
            restricted_region = (slice(x_trim_start, x_trim_end), slice(y_trim_start, y_trim_end), slice(None))
            
        # Apply the restriction
        restricted_normalized_array = normalized_array[restricted_region]
        restricted_dark_values_mask = restricted_normalized_array < np.percentile(restricted_normalized_array, percentile)
        
        # Label connected components in the restricted region
        structure = np.ones((3, 3, 3)) if connectivity == 2 else None
        labeled_array, num_features = label(restricted_dark_values_mask, structure=structure)

        if num_features == 0:
            print("No dark-value groups found.")
            return np.zeros_like(array_3d), 0, normalized_array, np.ones_like(array_3d)  # Return an empty mask and a modified array of ones if no groups are found

        # Find all group sizes by counting occurrences of each label
        label_counts = np.bincount(labeled_array.flat)

        # Exclude the background label (index 0) and sort groups by size in descending order
        sorted_labels_and_sizes = sorted(enumerate(label_counts[1:], start=1), key=lambda x: x[1], reverse=True)

        if rank > len(sorted_labels_and_sizes):
            print(f"Rank {rank} exceeds the number of detected groups. Returning an empty mask.")
            return np.zeros_like(array_3d), 0, normalized_array, np.ones_like(array_3d)  # Return an empty mask and a modified array of ones if rank is out of bounds

        # Get the label for the specified rank
        ranked_group_label, ranked_group_size = sorted_labels_and_sizes[rank - 1]

        # Create a mask for the ranked group in the restricted region
        ranked_group_mask_restricted = labeled_array == ranked_group_label

        # Expand the restricted mask to the original array shape
        ranked_group_mask = np.zeros_like(normalized_array, dtype=bool)
        ranked_group_mask[restricted_region] = ranked_group_mask_restricted

        # Modify the array: Keep only the values within the ranked group, set all others to 1
        modified_array = np.where(ranked_group_mask, array_3d, 1)

        return ranked_group_mask, modified_array, z_index

    def max_entropy_slice(
        array, 
        num_slices=10
        ):
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
    
    def find_points(
        array_3d, 
        slice_number, 
        num_slices=10, 
        threshold=0.9, 
        num_negative_points=5,
        num_positive_points=5,
        min_distance_percent=0.05, 
        negative_min_distance_percent=0.1
        ):
        """
        Finds the centroid of a black object (low-value pixels) in a 3D array by slicing along the shortest axis.
        Additionally, returns random positive points from the middle 40% region of the object and negative points 
        from non-black areas within the middle 60% of x and y dimensions, at a certain minimum distance from the mask.
        Ensures that negative points are far from each other.

        Parameters:
            array_3d (np.ndarray): The input 3D array.
            slice_number (int): The specific slice number to use (along the shortest axis).
            num_slices (int): Number of slices to average around the specified slice.
            threshold (float): The threshold below which values are considered "black".
            num_negative_points (int): Number of negative points to sample from non-black regions.
            num_positive_points (int): Number of random positive points to sample from the middle 40% of the object.
            min_distance_percent (float): Minimum distance as a percentage of the average size of the x and y dimensions from the black mask.
            negative_min_distance_percent (float): Minimum distance as a percentage of the average size of the x and y dimensions between negative points.

        Returns:
            positive_points (list): A list of positive point coordinates (including the centroid).
            negative_points (list): A list of random negative point coordinates from non-black areas.
        """
        # Determine the shortest axis
        shortest_axis = np.argmin(array_3d.shape)
        
        # Ensure the slice_number is within valid range
        slice_number = np.clip(slice_number, 0, array_3d.shape[shortest_axis] - 1)

        # Extract and average slices based on the shortest axis
        if shortest_axis == 0:
            start_slice = max(0, slice_number - num_slices // 2)
            end_slice = min(array_3d.shape[shortest_axis], slice_number + num_slices // 2 + 1)
            slices = array_3d[start_slice:end_slice, :, :]
            slice_2d = np.mean(slices, axis=0)
        elif shortest_axis == 1:
            start_slice = max(0, slice_number - num_slices // 2)
            end_slice = min(array_3d.shape[shortest_axis], slice_number + num_slices // 2 + 1)
            slices = array_3d[:, start_slice:end_slice, :]
            slice_2d = np.mean(slices, axis=1)
        elif shortest_axis == 2:
            start_slice = max(0, slice_number - num_slices // 2)
            end_slice = min(array_3d.shape[shortest_axis], slice_number + num_slices // 2 + 1)
            slices = array_3d[:, :, start_slice:end_slice]
            slice_2d = np.mean(slices, axis=2)
        
        # Create a mask where values below the threshold are considered "black"
        black_mask = slice_2d < threshold

        # If no black regions exist, return None
        if not black_mask.any():
            return None, None

        # Find the centroid of the black regions using the center of mass function
        centroid = center_of_mass(black_mask)
        
        # Convert (y, x) to (cx, cy) format
        centroid = (centroid[1], centroid[0])

        # ------------------ Positive Points (middle 40% of the object) ------------------

        # Get the bounding box of the black object
        black_coords = np.column_stack(np.where(black_mask))
        min_y, min_x = black_coords.min(axis=0)
        max_y, max_x = black_coords.max(axis=0)

        # Define the middle 40% of the object (bounding box)
        x_margin = 0.2 * (max_x - min_x)  # Exclude 20% from each side in x
        y_margin = 0.2 * (max_y - min_y)  # Exclude 20% from each side in y

        middle_40_mask = np.zeros_like(black_mask, dtype=bool)
        middle_40_mask[int(min_y + y_margin):int(max_y - y_margin), int(min_x + x_margin):int(max_x - x_margin)] = True

        # Combine the middle 40% mask with the black mask to get valid positive points
        valid_positive_mask = middle_40_mask & black_mask

        # Get coordinates of valid positive points within the middle 40%
        positive_coords = np.column_stack(np.where(valid_positive_mask))

        # If there are fewer points than requested, adjust the number
        if len(positive_coords) < num_positive_points:
            num_positive_points = len(positive_coords)

        # Randomly sample positive points from the valid coordinates
        if num_positive_points > 0:
            positive_points = positive_coords[np.random.choice(positive_coords.shape[0], num_positive_points, replace=False)]
        else:
            positive_points = np.array([])

        # Add the centroid as one of the positive points
        positive_points = np.vstack([np.array([[int(centroid[1]), int(centroid[0])]]), positive_points])

        # Convert (y, x) to (x, y) format
        positive_points = [(pt[1], pt[0]) for pt in positive_points]

        # ------------------ Negative Points (non-black areas in middle 60%) ------------------
        non_black_mask = ~black_mask

        # Middle 60% constraints for x and y dimensions
        x_start = int(black_mask.shape[1] * 0.2)
        x_end = int(black_mask.shape[1] * 0.8)
        y_start = int(black_mask.shape[0] * 0.2)
        y_end = int(black_mask.shape[0] * 0.8)

        # Create a mask for the middle 60% region
        middle_60_mask = np.zeros_like(non_black_mask, dtype=bool)
        middle_60_mask[y_start:y_end, x_start:x_end] = True

        # Combine middle 60% mask with non-black mask
        candidate_mask = non_black_mask & middle_60_mask

        # If no valid points exist, return positive_points and None
        if not candidate_mask.any():
            return positive_points, None

        # Calculate minimum distance in pixels based on the tomogram size
        avg_dimension_size = (black_mask.shape[1] + black_mask.shape[0]) / 2
        min_distance_pixels = avg_dimension_size * min_distance_percent
        negative_min_distance_pixels = avg_dimension_size * negative_min_distance_percent

        # Distance transform: get the distance of each point from the nearest black region
        distances = distance_transform_edt(non_black_mask)

        # Only consider points that are farther than the specified minimum distance from the black mask
        valid_distance_mask = distances >= min_distance_pixels
        final_candidate_mask = candidate_mask & valid_distance_mask

        # If no valid points exist after applying the distance constraint, return positive_points and None
        if not final_candidate_mask.any():
            return positive_points, None

        # Get coordinates of valid negative points
        negative_coords = np.column_stack(np.where(final_candidate_mask))

        # If there are not enough negative points, adjust the number
        if len(negative_coords) < num_negative_points:
            num_negative_points = len(negative_coords)

        # Initialize a list to hold selected negative points
        selected_negative_points = []

        # Randomly sample negative points ensuring a minimum distance between points
        np.random.shuffle(negative_coords)
        for candidate in negative_coords:
            if len(selected_negative_points) == 0:
                # Select the first point
                selected_negative_points.append(candidate)
            else:
                # Calculate the distance between the candidate and already selected points
                distances_to_selected = cdist([candidate], selected_negative_points)
                if np.all(distances_to_selected >= negative_min_distance_pixels):
                    # If the candidate is far enough from all selected points, add it to the list
                    selected_negative_points.append(candidate)
                    if len(selected_negative_points) >= num_negative_points:
                        break

        # Return the positive points and the negative points
        return positive_points, [(pt[1], pt[0]) for pt in selected_negative_points]  # Convert (y, x) to (x, y)

    def upscale_points(
        scaling_factor, 
        points
        ):
        """
        Upscales either a single 2D point or a list of 2D points using the provided scaling factor.

        Parameters:
        - scaling_factor (float): A single scalar value to scale both x and y dimensions.
        - points (tuple, list): A single point (x', y') or a list of 2D points to upscale.

        Returns:
        - Tuple or list of upscaled point(s).
        """
        # Convert points to a NumPy array
        points_array = np.array(points)

        # Check if it's a single point or multiple points
        if points_array.ndim == 1 and points_array.size == 2:
            # Single point
            upscaled_point = points_array * scaling_factor
            return tuple(upscaled_point)
        elif points_array.ndim == 2 and points_array.shape[1] == 2:
            # Multiple points
            upscaled_points = points_array * scaling_factor
            return [tuple(point) for point in upscaled_points]
        else:
            raise ValueError("Input must be either a single 2D point (x', y') or a list of 2D points.")

    def generate_images(array, save_path, max_frame_size=(512, 512), reflect_edges=False, num_slices=10):
        """
        Generates JPEG images by slicing across the shortest axis of a 3D NumPy array and saves them to disk.
        
        Parameters:
        - array: 3D numpy array to slice.
        - save_path: Directory path to save the JPEG images.
        - max_frame_size: Maximum size of each frame (width, height), used to resize large frames.
        - reflect_edges: Boolean flag to control whether to reflect the array at the edges.
        - num_slices: Number of slices to average around each frame (total window size is num_slices + 1).
        
        Returns:
        - resized_shape: Tuple (height, width) of the resized frames.
        """
        # Determine the shortest axis
        shortest_axis = np.argmin(array.shape)
        
        # Ensure the save_path directory exists
        os.makedirs(save_path, exist_ok=True)

        # Determine the original frame size based on the slicing axis
        height, width = (array.shape[1], array.shape[2]) if shortest_axis == 0 else \
                        (array.shape[0], array.shape[2]) if shortest_axis == 1 else \
                        (array.shape[0], array.shape[1])
        
        # Resize frames if they are too large
        if height > max_frame_size[1] or width > max_frame_size[0]:
            resize_factor_h = max_frame_size[1] / height
            resize_factor_w = max_frame_size[0] / width
            resize_factor = min(resize_factor_h, resize_factor_w)
            height = int(height * resize_factor)
            width = int(width * resize_factor)
        
        resized_shape = (height, width)
        
        # Reflect the array if the option is enabled
        if reflect_edges:
            pad_width = [(num_slices // 2, num_slices // 2) if i == shortest_axis else (0, 0) for i in range(3)]
            array = np.pad(array, pad_width, mode='reflect')
        
        # Iterate over slices along the shortest axis and save frames as images
        for i in range(array.shape[shortest_axis] - (num_slices if reflect_edges else 0)):
            # Calculate the start and end indices for averaging
            start_idx = max(0, i - num_slices // 2)
            end_idx = min(array.shape[shortest_axis], i + num_slices // 2 + 1)
            
            # Average the slices
            if shortest_axis == 0:
                slice_ = np.mean(array[start_idx:end_idx, :, :], axis=0)
            elif shortest_axis == 1:
                slice_ = np.mean(array[:, start_idx:end_idx, :], axis=1)
            else:
                slice_ = np.mean(array[:, :, start_idx:end_idx], axis=2)
            
            # Normalize the slice to 0-255 for image saving
            slice_normalized = cv2.normalize(slice_, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Resize the slice if necessary
            if height != slice_normalized.shape[0] or width != slice_normalized.shape[1]:
                slice_resized = cv2.resize(slice_normalized, (width, height), interpolation=cv2.INTER_AREA)
            else:
                slice_resized = slice_normalized
            
            # Convert to 3-channel grayscale for JPEG saving (JPEG expects 3 channels)
            slice_colored = cv2.cvtColor(slice_resized, cv2.COLOR_GRAY2BGR)
            
            # Flip the image vertically to fix orientation issues
            slice_flipped = cv2.flip(slice_colored, 0)
            
            # Save the frame as a JPEG image
            frame_path = os.path.join(save_path, f"{i:05d}.jpg")
            cv2.imwrite(frame_path, slice_flipped)

        print(f"JPEG frames saved in {save_path}")
        return resized_shape
            


    def rescale_coordinates(original_shape, 
                            resized_shape, 
                            coordinates
                            ):
        """
        Rescale coordinates from the original array shape to the resized image shape.

        Parameters:
        - original_shape: Tuple (height, width) of the original 2D slice.
        - resized_shape: Tuple (height, width) of the resized image.
        - coordinates: List of tuples [(x1, y1), (x2, y2), ...] representing the original coordinates.

        Returns:
        - rescaled_coordinates: List of tuples [(x1', y1'), (x2', y2'), ...] representing the rescaled coordinates.
        """
        original_height, original_width = original_shape
        resized_height, resized_width = resized_shape
        
        # print("Original Shape:", original_shape)
        # print("Resized Shape:", resized_shape)

        # Calculate scaling factors
        scale_x = resized_width / original_width
        scale_y = resized_height / original_height

        # print("Scale X:", scale_x)
        # print("Scale Y:", scale_y)  
        
        rescaled_coordinates = [
            (int(x * scale_x), int(y * scale_y)) for x, y in coordinates
        ]

        return rescaled_coordinates
    
    def slices_below_clustering(entropy_dict, n_clusters=2):
        # Prepare entropy data for clustering
        entropy_values = np.array(list(entropy_dict.values())).reshape(-1, 1)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(entropy_values)
        
        # Identify the cluster with the lowest mean entropy
        cluster_means = [np.mean(entropy_values[labels == i]) for i in range(n_clusters)]
        low_entropy_cluster = np.argmin(cluster_means)
        
        # Select slices in the low entropy cluster
        slices_below_threshold = {slice_num: entropy for slice_num, entropy in entropy_dict.items()
                                if labels[list(entropy_dict.keys()).index(slice_num)] == low_entropy_cluster}
        
        return slices_below_threshold

    def get_sequence_boundaries(entropy_dict):
        # Sort the keys
        sorted_keys = sorted(entropy_dict.keys())
        
        # Identify sequences of consecutive keys
        sequences = []
        current_sequence = [sorted_keys[0]]
        
        for i in range(1, len(sorted_keys)):
            if sorted_keys[i] == sorted_keys[i - 1] + 1:
                # Consecutive key, add to the current sequence
                current_sequence.append(sorted_keys[i])
            else:
                # New sequence starts, save the current sequence and start a new one
                sequences.append(current_sequence)
                current_sequence = [sorted_keys[i]]
        
        # Add the last sequence if it wasn't added
        if current_sequence:
            sequences.append(current_sequence)

        # Determine which sequences are "type 1" or "type 2" based on the first key in each sequence
        first_sequence_max = None
        second_sequence_min = None

        if sequences:
            # Check the first sequence type
            if sequences[0][0] <= 10:
                # "Type 1" sequence, get the largest key in the first sequence
                first_sequence_max = sequences[0][-1]
            else:
                # "Type 2" sequence, get the smallest key in the first sequence
                second_sequence_min = sequences[0][0]

            # Check if there is a second sequence
            if len(sequences) > 1:
                if second_sequence_min is None:
                    # Second sequence must be of "type 2" if it exists
                    second_sequence_min = sequences[1][0]
                else:
                    # If the first sequence was of type 1, the second sequence is of type 2
                    second_sequence_min = sequences[1][0]

        return first_sequence_max, second_sequence_min
    
    
    
    
    