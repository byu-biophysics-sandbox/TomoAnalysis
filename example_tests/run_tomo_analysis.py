from writers import CSV
from analyzers import TomoStats
from datasets import TomographyArray
from utils.data_factory import DataFactory
from image_processing_pipelines import ThresholdClusterPipeline
import os

#if "TOMOGRAM_PATH" not in os.environ.keys():
#    os.environ["TOMOGRAM_PATH"] = "/home/matiasgp/groups/grp_tomo_db1_d3/nobackup/archive/TomoDB1_d3/Hneptunium_secretin/aba2006-11-01-6/Hyphomonas_10bin_full.rec"

def main():
    # Sets up a tomography to pull slices from
    print("Compiling Dataset...")
    data = TomographyArray(["/home/matiasgp/groups/grp_tomo_db1_d1/nobackup/archive/TomoDB1_d1/FlagellarMotor_P1/Caulobacter crescentus"])
    
    # Sets up an analysis pipeline
    print("Setting up Analysis Pipeline...")
    analyzer = TomoStats()

    # Sets up a writer
    print("Setting up Writer...")
    writer = CSV("tomostats.csv")
    
    print("Configuring Processor...")
    batch_size = 1
    data_factory = DataFactory(
        analyzer,
        writer,
        batch_size
    )

    print("Processing Data...")
    data_factory.process(data, "Processing Slice")


if __name__ == "__main__":
    main()