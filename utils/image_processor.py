import numpy as np
from sklearn.cluster import DBSCAN

class ImageProcessor:
    @staticmethod
    def contrast(image, factor):
        mean = np.mean(image, axis=(0,1), keepdims=True)
        image_contrast = mean + factor * (image - mean)
        image_contrast = np.clip(image_contrast, 0.0, 1.0)
        return image_contrast
    
    @staticmethod
    def threshold_lt(
        image,
        threshold_factor: float,
    ):
        thresholded_pixels = image < threshold_factor
        mask = np.zeros_like(image)
        mask[thresholded_pixels] = 1
        return mask
    
    @staticmethod
    def  threshold_gt(
        image,
        threshold_factor: float,
    ):
        thresholded_pixels = image > threshold_factor
        mask = np.zeros_like(image)
        mask[thresholded_pixels] = 1
        return mask
    
    @staticmethod
    def cluster(
        x_points,
        y_points,
        **cluster_params
    ):
        clusterer = DBSCAN(**cluster_params)
        points = np.asarray(list(zip(x_points, y_points)))
        labels = clusterer.fit_predict(points)
        return labels
