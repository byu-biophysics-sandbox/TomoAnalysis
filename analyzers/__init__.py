from analyzers.analysis_base import Analysis
from analyzers.segmenter import BacteriaSegmenter
from analyzers.basic import Basic
from analyzers.tomo_stats import TomoStats
from analyzers.dataset_mean_analyzer import DatasetMeansAnalyzer
from analyzers.dataset_entropy_analyzer import DatasetEntropyAnalyzer
from analyzers.intensity_segmenter import IntensitySegmenter

def get_analysis(analysis_type):
    analysis_type = analysis_type.lower()

    if analysis_type == "basic":
        return Basic
    elif analysis_type == "segment":
        return BacteriaSegmenter
    else:
        raise Exception("Analysis Doesn't Exist")
