import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.transformers import ImageResizer
from onnxtr.models import linknet_resnet50
from onnxtr.models.detection import detection_predictor

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        
        text = ctc_decoder(preds, self.metadata["vocab"])[0]
    
        return text

class DetectionModel:
    def __init__(self, model_path: str):
        self.det_model = linknet_resnet50(model_path)
        self.detection = detection_predictor(arch=self.det_model)

    def predict(self, image: np.ndarray):
        return self.detection([image])