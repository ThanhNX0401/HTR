import os
import gdown
import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.transformers import ImageResizer
from onnxtr.models import linknet_resnet50
from onnxtr.models.detection import detection_predictor

class ImageToWordModel(OnnxInferenceModel):
    model_downloaded = False

    def __init__(self, model_path: str, *args, **kwargs):
        if not ImageToWordModel.model_downloaded:
            gdown.download("https://drive.google.com/uc?id=1I7TAd8_C7xRzgzMSs6cNvMjUzwB8h2Ny", model_path, quiet=False)
            ImageToWordModel.model_downloaded = True

        super().__init__(model_path=model_path, *args, **kwargs)

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        
        text = ctc_decoder(preds, self.metadata["vocab"])[0]
    
        return text

class DetectionModel:
    model_downloaded = False

    def __init__(self, model_path: str):
        if not DetectionModel.model_downloaded:
            gdown.download("https://drive.google.com/uc?id=1ZJoys_gCy9GzbFgo5L5bk8kO5z5nwexu", model_path, quiet=False)
            DetectionModel.model_downloaded = True

        self.det_model = linknet_resnet50(model_path)
        self.detection = detection_predictor(arch=self.det_model)

    def predict(self, image: np.ndarray):
        return self.detection([image])