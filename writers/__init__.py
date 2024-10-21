from writers.writer_base import Writer
from writers.csv import CSV
from writers.excel import Excel
from writers.tomo_seg import SegmentationSaver
from writers.video_plotter import Video
from writers.img_plotter import Slide
from writers.dict import DictSaver


def get_writer(
        writer_string: str = "csv",
        output_folder: str = None
    ):
    writer_string = writer_string.lower()
    
    if writer_string == "csv":
        return CSV
    elif writer_string == "xlsx":
        return Excel
    elif writer_string == "seg":
        return SegmentationSaver
    else:
        raise Exception("Writer does not exist")