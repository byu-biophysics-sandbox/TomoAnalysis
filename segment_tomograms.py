from writers import SegmentationSaver
from analyzers import BacteriaSegmenter
from datasets import TomographyArray
from utils.data_factory import DataFactory
from image_processing_pipelines import ThresholdClusterPipeline
import os

if "TOMOGRAM_PATH" not in os.environ.keys():
    os.environ["TOMOGRAM_PATH"] = "/home/matiasgp/groups/grp_tomo_db1_d3/nobackup/archive/TomoDB1_d3/"

def main():
    # Sets up a tomography to pull slices from
    print("Compiling Dataset...")
    data = TomographyArray(os.environ["TOMOGRAM_PATH"])

    print("Setting up Analysis Pipeline...")
    # Sets up a segmentation pipeline
    device = "cuda"
    analyzer = BacteriaSegmenter(
        image_processing_pipeline=ThresholdClusterPipeline(),
        device=device
    )

    print("Setting up Writer...")
    writer = SegmentationSaver(data.bacteria_name)
    
    print("Configuring Processor...")
    batch_size = 2
    data_factory = DataFactory(
        analyzer,
        writer,
        batch_size
    )

    print("Processing Data...")
    data_factory.process(data, "Processing Slice")


if __name__ == "__main__":
    main()