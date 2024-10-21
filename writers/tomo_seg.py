from writers import Writer
import matplotlib.pyplot as plt
import os
from utils.graph_plotter import GraphPlotter
import cv2
import h5py
import numpy as np

class SegmentationSaver(Writer):
    def __init__(
            self,
            bacteria_name,
            output_file="./segmentation_images/",
            alpha=0.6
        ):
        super().__init__(output_file)
        self.bacteria_name = bacteria_name
        self.bacteria_folder = os.path.join(self.file_path, bacteria_name)
        if not os.path.exists(self.bacteria_folder):
            os.makedirs(self.bacteria_folder)
        
        self.computed_segmentations = {}
        self.alpha = alpha
    
    def write(self, results: dict, **kwargs):
        self.computed_segmentations.update(results)

        slice_filepath = os.path.join(self.bacteria_folder, "slices")
        if not os.path.exists(slice_filepath):
            os.makedirs(slice_filepath)

        for key, value in results.items():
            fig, axes = plt.subplots(1, 3)

            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])


            axes[0].set_title("Original")
            axes[0].imshow(value["image"], cmap="gray")

            axes[1].set_title("Segmentation")
            axes[1].imshow(value["image"], cmap="gray")
            GraphPlotter.show_mask(value["mask"], axes[1])

            axes[2].imshow(value["image"], cmap="gray")
            axes[2].set_title("Seg W/ Points")
            GraphPlotter.show_mask(value["mask"], axes[2])
            GraphPlotter.show_points(value["pts"]["coords"], value["pts"]["labels"], axes[2])

            fig.suptitle(f"{self.bacteria_name} Segmentation Slice Idx: {key}", y=.8, fontsize=15)
            fig.tight_layout()

            plt.savefig(os.path.join(slice_filepath, f"{self.bacteria_name}_seg_{key}.png"))
            plt.close(fig)

    def finish_and_save(self):
        self.segmentations = []
        for key, value in self.computed_segmentations.items():
            self.segmentations.append((key, value["mask"], value["image"]))
        self.segmentations.sort(lambda x:x[0])

        self.__create_segmentation_video()
        self.__save_mask_array()

    def __create_segmentation_video(self):
        width, height = self.segmentations[0][1].shape

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_file = os.path.join(self.bacteria_folder, "segmentation_video.mp4")
        out = cv2.VideoWriter(video_file, fourcc, 1.0, (width, height))

        for index, mask, image in self.segmentations:
            overlay = image * (1 - self.alpha * mask) + (self.alpha * mask * 255)
            overlay = overlay.astype(np.uint8)
            out.write(overlay)
        
        out.release()
        cv2.destroyAllWindows()
    
    def __save_mask_array(self):
        masks = []
        for _, mask, _ in self.segmentations:
            masks.append(mask)
        tomogram_stack = np.vstack(masks)

        with h5py.File('example.jld2', 'w') as f:
            f.create_dataset("mask", data=tomogram_stack)
        
    
        
