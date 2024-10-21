import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from .img_plotter import Slide
import numpy as np
import imageio as iio
import os


class Video(Slide):
    def __init__(self, 
                 numpy_array, 
                 output_name
                 ):
        """
        Initialize the Video object by calling the parent Slide class constructor.
        
        Parameters:
        numpy_array (numpy.ndarray): The 3D numpy array representing the image data.
        output_name (str): The name of the output video file.
        """
        super().__init__(numpy_array, output_name)
        
    def plot_video(self, 
                   initial_slice, 
                   final_slice, 
                   slices_per_frame, 
                   axis
                   ):
        """
        Create a video from a series of 2D slices of the 3D numpy array.
        
        Parameters:
        initial_slice (int): The starting index for slicing.
        final_slice (int): The ending index for slicing.
        slices_per_frame (int): The number of slices to average per frame.
        axis (int): The axis along which to take the slices (0, 1, or 2).
        """
        filenames = []  # List to store temporary filenames
        listofmeans = [] # List to store temporary means

        for start in range(initial_slice, final_slice, slices_per_frame):
            end = min(start + slices_per_frame, final_slice)
            xlabel, ylabel = "", ""
            
            # Calculate the average of the slices along the specified axis
            if axis == 0:
                slice_data = np.mean(self.numpy_array[start:end, :, :], axis=0)
                xlabel, ylabel = 'Y axis', 'Z axis'
            elif axis == 1:
                slice_data = np.mean(self.numpy_array[:, start:end, :], axis=1)
                xlabel, ylabel = 'Z axis', 'X axis'
            elif axis == 2:
                slice_data = np.mean(self.numpy_array[:, :, start:end], axis=2)
                xlabel, ylabel = 'Y axis', 'X axis'
            else:
                raise ValueError("Axis must be 0, 1, or 2")
            
            # Create a temporary file to save the plot
            temp_filename = f'temp_slice_{start}_{end}_{self.outputname+str(axis)}.png'
            temp = Slide(slice_data, temp_filename)
            listofmeans.append(temp.plot_2d_img(axis, start, xlabel, ylabel))
            filenames.append(temp_filename)
        
        # Define the path for the video file
        output_directory = '/home/matiasgp/Desktop/Toma-Analysis/' + self.outputname + "/"
        output_filename =  self.outputname + str(axis) + '.mp4'
        video_path = os.path.join(output_directory, output_filename)

        # Ensure the output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Create the video from the saved images
        with iio.get_writer(video_path, mode='I', fps=4) as writer:
            print("Creating Video")
            for filename in filenames:
                image = iio.imread(filename)
                writer.append_data(image)
                
        # Create a plot with the means through time
        plot_bar_chart(self,
                       data=listofmeans,
                       axis=axis,
                       title=self.outputname + str(axis),
                       ylabel="Mean"
                       )
        
        # Clean up temporary files
        for filename in filenames:
            os.remove(filename)

# Comments added by ChatGPT
def plot_bar_chart(self,
                   data,
                   axis, 
                   title='Bar Chart',
                   xlabel='Slice',
                   ylabel='Value'
                   ):
    """
    Plots a bar chart of the given list of numbers and saves it with the title as the filename.

    Parameters:
    - data: list of numbers to plot the bar chart for.
    - title: title of the bar chart (default is 'Bar Chart').
    - xlabel: label for the x-axis (default is 'Index').
    - ylabel: label for the y-axis (default is 'Value').
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(data)), data, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Define the directory and filename
    output_directory = '/home/matiasgp/Desktop/Toma-Analysis/' + self.outputname + "/"
    filename = os.path.join(output_directory, "hist_" + title.replace(' ', '_') + '.png')
    
    # Create the directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Save the plot to the specified file
    plt.savefig(filename)
    plt.close()
    
