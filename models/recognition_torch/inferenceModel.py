import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.metadata["vocab"])[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    model = ImageToWordModel(model_path="Models/08_handwriting_recognition_torch/202303142139/model.onnx")

    df = pd.read_csv("Models/08_handwriting_recognition_torch/202303142139/val.csv").values.tolist()

    accum_cer = []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

        accum_cer.append(cer)

    print(f"Average CER: {np.average(accum_cer)}")
    
# {'b', ':', 'ụ', 'í', 'K', 'ữ', 'ă', '+', 'à', 'x', '1', 'Ư', 'O', '@', 'Ý', '%', '=', 'Ọ', 'ự', 'Ĩ', 'j', '−', 'ừ', 'Ă', '/', 'z', 'Ù', 'Ừ', 'Ạ', 'ô', ',', 'ấ', 'a', '[', 'N', 'M', 'Ò', 'n', 'E', 'ẽ', 'ặ', 'ê', '’', 'Ợ', 'ở', 'A', 'ể', '5', 'Y', 'ậ', 'v', 'ƒ', 'Q', '∗', '$', 'Ỹ', 'ỉ', '“', 'u', 'ư', 'y', 'é', 'ý', 'P', '”', 'Ắ', 'Ổ', 'X', 'h', 'U', '-', 'Ứ', 'ẫ', '^', '{', 'ò', 'e', '~', 'ờ', 'ề', 'f', 'W', '"', 'è', 'Ặ', '£', 'Ụ', 'w', 'ằ', 'ẳ', 'Ể', 'Ự', 'Ả', 'd', '0', 'Ằ', 'V', 'Ẻ', '.', 'ế', '<', 'Ữ', 'Đ', 'Ầ', 'q', '*', 'F', 'ạ', 'Ẩ', 'Ẹ', 'ĩ', '?', 'Ơ', 'C', 'Ủ', 'Ỳ', 'ả', '&', '‘', 'Ử', '4', 'Ộ', 'Ỷ', '>', '}', 'D', 'Ỵ', 'c', 'ẵ', '(', 'H', 'ọ', 'l', '8', 'Z', 'Ế', "'", 'Ỉ', 'ủ', 'T', 'Ó', 'Ờ', 'Á', 'I', 'Ớ', 'Ỗ', 'Ị', 't', 'p', 'ổ', 'É', '!', 'À', 'Ẫ', 'Ỏ', 'ỳ', 'Ũ', 'Ẳ', '_', 'R', 'r', '6', 'Ố', 'k', 'ũ', 'Í', '\u200b', 'ộ', 'ì', 'ỡ', 'Ậ', 'ỏ', 'Ệ', 'ỷ', 'ó', 'ố', 'ù', 'J', 'Ì', 'Ô', '2', 'ẻ', 'â', 'Ở', ']', 'ễ', 'Ỡ', 'B', 'G', 'Ã', '#', 'i', 'ợ', 'L', ')', ';', 'ỗ', 'Ồ', 'ẹ', 'ứ', 'm', 'ử', '7', 's', 'á', 'ú', 'đ', 'ỹ', 'ẩ', 'Õ', 'Ẵ', 'Ấ', 'Â', '…', '9', 'õ', 'g', 'Ề', 'ơ', 'ắ', 'ỵ', 'Ú', 'ệ', 'S', 'ầ', 'ớ', 'Ê', '3', 'Ễ', 'È', 'ị', 'o', 'ồ', 'Ẽ', 'ã', '|'}