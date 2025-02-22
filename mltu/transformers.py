import cv2
import time
import queue
import typing
import logging
import threading
import importlib
import numpy as np

from . import Image
from scipy.ndimage import interpolation as inter
from mltu.annotations.detections import Detections

def correct_skew(image: np.ndarray, delta: int = 1, limit: int = 20) -> typing.Tuple[float, np.ndarray]:
    """Corrects skew in the image by detecting the best rotation angle."""
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated


class Transformer:
    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def __call__(self, data: typing.Any, label: typing.Any, *args, **kwargs):
        raise NotImplementedError


class ExpandDims(Transformer):
    def __init__(self, axis: int=-1):
        self.axis = axis

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return np.expand_dims(data, self.axis), label

class ImageResizer(Transformer):
    """Resize image to (width, height)
    
    Attributes:
        width (int): Width of image
        height (int): Height of image
        keep_aspect_ratio (bool): Whether to keep aspect ratio of image
        padding_color (typing.Tuple[int]): Color to pad image
    """
    def __init__(
        self, 
        width: int, 
        height: int, 
        keep_aspect_ratio: bool=False, 
        padding_color: typing.Tuple[int]=(0, 0, 0)
        ) -> None:
        self._width = width
        self._height = height
        self._keep_aspect_ratio = keep_aspect_ratio
        self._padding_color = padding_color

    @staticmethod
    def unpad_maintaining_aspect_ratio(padded_image: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
        height, width = padded_image.shape[:2]
        ratio = min(width / original_width, height / original_height)

        delta_w = width - int(original_width * ratio)
        delta_h = height - int(original_height * ratio)
        left, right = delta_w//2, delta_w-(delta_w//2)
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        unpaded_image = padded_image[top:height-bottom, left:width-right]

        original_image = cv2.resize(unpaded_image, (original_width, original_height))

        return original_image

    @staticmethod
    def resize_maintaining_aspect_ratio(image: np.ndarray, width_target: int, height_target: int, padding_color: typing.Tuple[int]=(0, 0, 0)) -> np.ndarray:
        """ Resize image maintaining aspect ratio and pad with padding_color.

        Args:
            image (np.ndarray): Image to resize
            width_target (int): Target width
            height_target (int): Target height
            padding_color (typing.Tuple[int]): Color to pad image

        Returns:
            np.ndarray: Resized image
        """
        _, corrected_image = correct_skew(image)

        # Convert to grayscale
        gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, d=3, sigmaColor=75, sigmaSpace=75)

        # Apply threshold and invert (black foreground)
        _, binary_mask = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted_binary = cv2.bitwise_not(binary_mask)  # Invert to have black as foreground

        # Apply dilation to the inverted binary mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        dilated_image = cv2.dilate(inverted_binary, kernel, iterations=1)
        dilated_image = 255 - dilated_image
        
        # Apply morphological operations (open) on the dilated image
        # morph_image = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Create a binary image with 3 channels
        binary_image = np.stack((dilated_image,) * 3, axis=-1)
        height, width = corrected_image.shape[:2]
        ratio = min(width_target / width, height_target / height)
        new_w, new_h = int(width * ratio), int(height * ratio)

        # Resize the image
        resized_image = cv2.resize(binary_image, (new_w, new_h))
        delta_w = width_target - new_w
        delta_h = height_target - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

        return new_image

    def __call__(self, image: Image, label: typing.Any) -> typing.Tuple[Image, typing.Any]:
        if not isinstance(image, Image):
            raise TypeError(f"Expected image to be of type Image, got {type(image)}")

        # Maintains aspect ratio and resizes with padding.
        if self._keep_aspect_ratio:
            image_numpy = self.resize_maintaining_aspect_ratio(image.numpy(), self._width, self._height, self._padding_color)
            if isinstance(label, Image):
                label_numpy = self.resize_maintaining_aspect_ratio(label.numpy(), self._width, self._height, self._padding_color)
                label.update(label_numpy)
        else:   
            # Resizes without maintaining aspect ratio.
            image_numpy = cv2.resize(image.numpy(), (self._width, self._height))
            if isinstance(label, Image):
                label_numpy = cv2.resize(label.numpy(), (self._width, self._height))
                label.update(label_numpy)

        image.update(image_numpy)

        return image, label

class LabelIndexer(Transformer):
    """Convert label to index by vocab
    
    Attributes:
        vocab (typing.List[str]): List of characters in vocab
    """
    def __init__(
        self, 
        vocab: typing.List[str]
        ) -> None:
        self.vocab = vocab

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])

class LabelPadding(Transformer):
    """Pad label to max_word_length
    
    Attributes:
        padding_value (int): Value to pad
        max_word_length (int): Maximum length of label
        use_on_batch (bool): Whether to use on batch. Default: False
    """
    def __init__(
        self, 
        padding_value: int,
        max_word_length: int = None, 
        use_on_batch: bool = False
        ) -> None:
        self.max_word_length = max_word_length
        self.padding_value = padding_value
        self.use_on_batch = use_on_batch

        if not use_on_batch and max_word_length is None:
            raise ValueError("max_word_length must be specified if use_on_batch is False")

    def __call__(self, data: np.ndarray, label: np.ndarray):
        if self.use_on_batch:
            max_len = max([len(a) for a in label])
            padded_labels = []
            for l in label:
                padded_label = np.pad(l, (0, max_len - len(l)), "constant", constant_values=self.padding_value)
                padded_labels.append(padded_label)

            padded_labels = np.array(padded_labels)
            return data, padded_labels

        label = label[:self.max_word_length]
        return data, np.pad(label, (0, self.max_word_length - len(label)), "constant", constant_values=self.padding_value)


class ImageNormalizer:
    """ Normalize image to float value, transpose axis if necessary and convert to numpy
    """
    def __init__(self, transpose_axis: bool=False):
        """ Initialize ImageNormalizer

        Args:
            transpose_axis (bool): Whether to transpose axis. Default: False
        """
        self.transpose_axis = transpose_axis
    
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[np.ndarray, typing.Any]:
        """ Convert each Image to numpy, transpose axis ant normalize to float value
        """
        img = image.numpy() / 255.0

        if self.transpose_axis:
            img = img.transpose(2, 0, 1)
        
        return img, annotation


class ImageShowCV2(Transformer):
    """Show image for visual inspection
    """
    def __init__(
        self, 
        verbose: bool = True,
        log_level: int = logging.INFO,
        name: str = "Image"
        ) -> None:
        """
        Args:
            verbose (bool): Whether to log label
            log_level (int): Logging level (default: logging.INFO)
            name (str): Name of window to show image
        """
        super(ImageShowCV2, self).__init__(log_level=log_level)
        self.verbose = verbose
        self.name = name
        self.thread_started = False

    def init_thread(self):
        if not self.thread_started:
            self.thread_started = True
            self.image_queue = queue.Queue()

            # Start a new thread to display the images, so that the main loop could run in multiple threads
            self.thread = threading.Thread(target=self._display_images)
            self.thread.start()

    def _display_images(self) -> None:
        """ Display images in a continuous loop """
        while True:
            image, label = self.image_queue.get()
            if isinstance(label, Image):
                cv2.imshow(self.name + "Label", label.numpy())
            cv2.imshow(self.name, image.numpy())
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def __call__(self, image: Image, label: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Show image for visual inspection

        Args:
            data (np.ndarray): Image data
            label (np.ndarray): Label data
        
        Returns:
            data (np.ndarray): Image data
            label (np.ndarray): Label data (unchanged)
        """
        # Start cv2 image display thread
        self.init_thread()

        if self.verbose:
            if isinstance(label, (str, int, float)):
                self.logger.info(f"Label: {label}")

        if isinstance(label, Detections):
            for detection in label:
                img = detection.applyToFrame(np.asarray(image.numpy()))
                image.update(img)

        # Add image to display queue
        # Sleep if queue is not empty
        while not self.image_queue.empty():
            time.sleep(0.5)

        # Add image to display queue
        self.image_queue.put((image, label))

        return image, label

class ImageBinarizer(Transformer):
    """Binarize the input image using morphological operations and Otsu's thresholding.
    
    Attributes:
        structuring_element_size (tuple): Size of the structuring element for morphological operations.
    """
    def __init__(self, structuring_element_size: tuple = (8, 8)) -> None:
        self.structuring_element_size = structuring_element_size

    def __call__(self, image: Image, label: typing.Any) -> typing.Tuple[Image, typing.Any]:
        if not isinstance(image, Image):
            raise TypeError(f"Expected image to be of type Image, got {type(image)}")

        # Convert the image to a numpy array and to grayscale
        image_numpy = image.numpy()
        gray = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2GRAY)
        
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Create a binary image with 3 channels
        binary_image = np.stack((binary_mask,) * 3, axis=-1)  # Stack the binary mask to create 3 channels

        # Update the image with the binarized result
        image.update(binary_image)

        # Create a structuring element for morphological operations
        # se = cv2.getStructuringElement(cv2.MORPH_RECT, self.structuring_element_size)

        # # Apply dilation
        # bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, se)

        # # Divide the original image by the dilated background
        # out_gray = cv2.divide(gray, bg, scale=255)

        
        # Apply Otsu's thresholding
        # _, out_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        # binarized_image = Image.fromarray(out_binary)
        # Update the image with the binarized result
        # image.update(out_binary)
        
        
        
        return image, label