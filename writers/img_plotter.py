import matplotlib.pyplot as plt
import numpy as np

class Slide:
    def __init__(self, 
                 numpy_array, 
                 output_name
                 ):
        """
        Initialize the Slide object.
        
        Parameters:
        numpy_array (numpy.ndarray): The 3D numpy array representing the image data.
        output_name (str): The name of the output file where plots will be saved.
        """
        self.numpy_array = numpy_array  # Store the numpy array
        self.outputname = output_name  # Store the output file name
        
    def plot_2d_img(self, 
                    axis, 
                    slice_index, 
                    xlabel, 
                    ylabel
                    ):
        """
        Plot a 2D slice of the 3D image array and save it to the output file.
        
        Parameters:
        axis (int): The axis along which to take the slice (0, 1, or 2).
        slice_index (int): The index of the slice along the specified axis.
        xlabel (str): The label for the x-axis of the plot.
        ylabel (str): The label for the y-axis of the plot.
        """
        plt.imshow(self.numpy_array, cmap='gray')   # Display the numpy array as an image
        plt.colorbar()  # Add a colorbar to the image
        plt.title(f'Slice along axis {axis} at index {slice_index}')  # Add a title to the plot
        plt.xlabel(xlabel)  # Set the x-axis label
        plt.ylabel(ylabel)  # Set the y-axis label
        
        # Calculate the mean value of the array
        mean_value = np.mean(self.numpy_array)
        
        # Add mean value annotation to the top right corner of the plot
        plt.text(0.95, 0.95, f'Mean: {mean_value:.2f}', color='white', 
                 fontsize=12, ha='right', va='top', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))
        
        # Save the plot to the specified output file
        plt.savefig(self.outputname)
        plt.close()
        return mean_value
    
    def plot_3d_img(self,
                    axis, 
                    slice_index
                    ):
        """
        Plot a 2D slice from the 3D image array along a specified axis and save it to the output file.
        
        Parameters:
        axis (int): The axis along which to take the slice (0, 1, or 2).
        slice_index (int): The index of the slice along the specified axis.
        """
        # Determine the slice data and axis labels based on the specified axis
        if axis == 0:
            slice_data = self.numpy_array[slice_index, :, :]
            xlabel, ylabel = 'Y axis', 'Z axis'
        elif axis == 1:
            slice_data = self.numpy_array[:, slice_index, :]
            xlabel, ylabel = 'X axis', 'Z axis'
        elif axis == 2:
            slice_data = self.numpy_array[:, :, slice_index]
            xlabel, ylabel = 'X axis', 'Y axis'
        else:
            raise ValueError("Axis must be 0, 1, or 2")
        
        # Calculate the mean value of the slice
        mean_value = np.mean(slice_data)

        plt.imshow(slice_data, cmap='viridis')  # Display the slice as an image
        plt.gray()  # Apply a gray colormap
        plt.title(f'Slice along axis {axis} at index {slice_index}')  # Add a title to the plot
        plt.xlabel(xlabel)  # Set the x-axis label
        plt.ylabel(ylabel)  # Set the y-axis label
        
        # Add mean value annotation to the top right corner of the plot
        plt.text(0.95, 0.95, f'Mean: {mean_value:.2f}', color='white', 
                 fontsize=12, ha='right', va='top', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))
        
        # Save the plot to the specified output file
        plt.savefig(self.outputname)
        plt.close()

# Comments added by ChatGPT
