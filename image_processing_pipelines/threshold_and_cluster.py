from image_processing_pipelines.image_pipeline import ImagePipeline
from utils.image_processor import ImageProcessor
import cv2
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

class ThresholdClusterPipeline(ImagePipeline):
    def __init__(
            self,
            contrast_factor:float=8,
            threshold_factor:float=0.01,
            dbscan_epsilon:float=5,
            gaussian_sigma:float=200,
            display_plots:bool=False,
        ):
        self.contrast_factor = contrast_factor
        self.threshold_factor = threshold_factor
        self.dbscan_epsilon = dbscan_epsilon
        self.gaussian_sigma = gaussian_sigma
        self.display_plots = display_plots
    
    def process_image(
            self,
            image,
            **kwargs
        ):
        
        contrasted_bacteria = ImageProcessor.contrast(image, self.contrast_factor)
        bacteria_mask = ImageProcessor.threshold_lt(contrasted_bacteria, self.threshold_factor)
        if self.display_plots:
            plt.imshow(bacteria_mask, cmap="gray")
            plt.title("Contrast Bacteria")
            plt.show()

        blurred_bacteria = cv2.GaussianBlur(contrasted_bacteria, (15,15), self.gaussian_sigma)
        if self.display_plots:
            plt.imshow(blurred_bacteria, cmap="gray")
            plt.title("Blurred Contrast Bacteria")
            plt.show()


        # Create filtered mask
        initial_detection = bacteria_mask.copy()
        blurred_mask = cv2.blur(initial_detection,(15,15))
        second_bacteria_mask = ImageProcessor.threshold_gt(blurred_mask, np.percentile(blurred_mask, 95))

        if self.display_plots:
            plt.imshow(second_bacteria_mask, cmap="gray")
            plt.show()

        # Cluster filtered mask to get clustered centroids
        initial_points = np.where(second_bacteria_mask == 1)
        labels = ImageProcessor.cluster(
            x_points=initial_points[0],
            y_points=initial_points[1],
            eps=self.dbscan_epsilon,
            min_samples=20
        )

        # Format the points in a specific way
        point_classes = np.concatenate([np.array(initial_points), labels.reshape(1,-1)]).transpose(1,0)
        df = pd.DataFrame(point_classes, columns=["x_coord", "y_coord", "labels"])
        centroids = df.groupby("labels").mean()
        centroids.drop(-1, axis=0)
        centroids = centroids.to_numpy()

        if self.display_plots:
            plt.imshow(image, cmap="gray")
            plt.scatter(initial_points[1], initial_points[0], c=labels)
            cx, cy = zip(*centroids)
            plt.scatter(cy, cx, c="red")
            plt.title("Points and Clusters")
            plt.show()

        # Cluster the initial centroids
        dbscan_two = DBSCAN(
            eps=200,
            min_samples=20
        )
        centroids = centroids
        labels = dbscan_two.fit_predict(centroids)

        # Find the cluster with the largest number of centroids.
        counts = {str(label): [] for label in set(labels)}
        for i in range(len(labels)):
            label = str(labels[i])
            counts[label].append(centroids[i][::-1])
        max_count = -1
        max_key = 0
        for cluster_key, value in counts.items():
            if len(value) > max_count:
                max_key = cluster_key
                max_count = len(value)
        
        predicted_bacteria_locations = np.array(counts[max_key])

        return {
            "positive_points": predicted_bacteria_locations,
            "negative_points": predicted_bacteria_locations,
            "bacteria_to_segment": blurred_bacteria
        }