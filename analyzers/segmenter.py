from image_processing_pipelines.image_pipeline import ImagePipeline
import numpy as np
from analyzers import Analysis
import os

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import torch
import warnings
import onnxruntime

class BacteriaSegmenter(Analysis):
    def __init__(
        self,
        image_processing_pipeline: ImagePipeline,
        sam_checkpoint = "sam_vit_h_4b8939.pth",
        model_type = "vit_h",
        device="cuda",
        model_path:str="./models/"
    ):
        self.ipp = image_processing_pipeline
        self.model_path = model_path

        if not os.path.exists(model_path):
            os.makedirs(model_path)
            self.model_path = model_path

        self.__setup_sam(
            sam_checkpoint=sam_checkpoint,
            model_type=model_type,
            device=device
        )
        
   
    def analyze(
            self,
            data,
            key,
            results: dict,
            **kwargs
        ):

        # Process Images
        processed_data = self.ipp.process_image(data)

        # Format data to be usable by SAM
        positive_labels = np.ones_like(processed_data["positive_points"][:,0]).astype(int)
        negative_labels = np.zeros_like(processed_data["negatives_points"][:,0]).astype(int)

        points = np.concatenate([processed_data["positive_points"], processed_data["negatives_points"]])
        labels = np.concatenate([positive_labels, negative_labels])
        bacteria = np.concatenate([processed_data["bacteria_to_segment"][np.newaxis, ...]] * 3, axis=0).transpose((1,2,0))

        # Run SAM
        mask = self.__run_sam(
            points,
            labels,
            bacteria,
        )

        seg_results = {
            "image": data,
            "mask": mask,
            "pts": {
                "coords": points,
                "labels": labels
            }
        }

        results[key] = seg_results
    
    def __setup_sam(
            self,
            model_type,
            sam_checkpoint,
            device,
            onnx_filename="sam_onnx.onnx"
        ):
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        self.onnx_filepath = os.path.join(self.model_path, onnx_filename)
        if not os.path.exists(self.onnx_filepath):
            onnx_model_path = "sam_onnx_example.onnx"

            onnx_model = SamOnnxModel(self.sam, return_single_mask=True)
            dynamic_axes = {
                "point_coords": {1: "num_points"},
                "point_labels": {1: "num_points"},
            }

            embed_dim = self.sam.prompt_encoder.embed_dim
            embed_size = self.sam.prompt_encoder.image_embedding_size
            mask_input_size = [4 * x for x in embed_size]
            dummy_inputs = {
                "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
                "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
                "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
                "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
                "has_mask_input": torch.tensor([1], dtype=torch.float),
                "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
            }
            output_names = ["masks", "iou_predictions", "low_res_masks"]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                with open(onnx_model_path, "wb") as f:
                    torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        f,
                        export_params=True,
                        verbose=False,
                        opset_version=17,
                        do_constant_folding=True,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                    )   
        
        self.sam_onnx = SamOnnxModel(self.sam, return_single_mask=True)
        self.sam.to(device)
    
    def __run_sam(
            self,
            bacteria,
            coords,
            labels,
        ):
        predictor = SamPredictor(self.sam)
        predictor.set_image(bacteria)
        image_embedding = predictor.get_image_embedding().cpu().numpy()

        ort_session = onnxruntime.InferenceSession(self.onnx_filepath)
        onnx_coord = np.concatenate([coords, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
        onnx_coord = predictor.transform.apply_coords(onnx_coord, bacteria[:2]).astype(np.float32)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(bacteria.shape[:2], dtype=np.float32)
        }

        mask, _, _ = ort_session.run(None, ort_inputs)
        mask = mask > predictor.model.mask_threshold

        return mask
