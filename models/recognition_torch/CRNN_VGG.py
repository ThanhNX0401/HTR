
#45
import string

__all__ = ["VOCABS"]

VOCABS: dict[str, str] = {
    "vietnamese": (
        string.digits + string.ascii_letters + string.punctuation
        + "áàảạãăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệóòỏõọôốồổộỗơớờởợỡúùủũụưứừửữựíìỉĩịýỳỷỹỵ"
        + "ÁÀẢẠÃĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỘỖƠỚỜỞỢỠÚÙỦŨỤƯỨỪỬỮỰÍÌỈĨỊÝỲỶỸỴ"
    )
}

# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import string
import unicodedata
from collections.abc import Sequence
from collections.abc import Sequence as SequenceType
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from PIL import Image

from doctr.io.image import get_img_shape
from doctr.utils.geometry import convert_to_relative_coords, extract_crops, extract_rcrops

from .vocabs import VOCABS

__all__ = [
    "translate",
    "encode_string",
    "decode_sequence",
    "encode_sequences",
    "pre_transform_multiclass",
    "crop_bboxes_from_image",
    "convert_target_to_relative",
]

ImageTensor = TypeVar("ImageTensor")


def translate(
    input_string: str,
    vocab_name: str,
    unknown_char: str = "■",
) -> str:
    """Translate a string input in a given vocabulary

    Args:
        input_string: input string to translate
        vocab_name: vocabulary to use (french, latin, ...)
        unknown_char: unknown character for non-translatable characters

    Returns:
        A string translated in a given vocab
    """
    if VOCABS.get(vocab_name) is None:
        raise KeyError("output vocabulary must be in vocabs dictionnary")

    translated = ""
    for char in input_string:
        if char not in VOCABS[vocab_name]:
            # we need to translate char into a vocab char
            if char in string.whitespace:
                # remove whitespaces
                continue
            # normalize character if it is not in vocab
            char = unicodedata.normalize("NFD", char).encode("ascii", "ignore").decode("ascii")
            if char == "" or char not in VOCABS[vocab_name]:
                # if normalization fails or char still not in vocab, return unknown character)
                char = unknown_char
        translated += char
    return translated


def encode_string(
    input_string: str,
    vocab: str,
) -> list[int]:
    """Given a predefined mapping, encode the string to a sequence of numbers

    Args:
        input_string: string to encode
        vocab: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
        A list encoding the input_string
    """
    try:
        return list(map(vocab.index, input_string))
    except ValueError:
        raise ValueError(
            f"some characters cannot be found in 'vocab'. \
                         Please check the input string {input_string} and the vocabulary {vocab}"
        )


def decode_sequence(
    input_seq: np.ndarray | SequenceType[int],
    mapping: str,
) -> str:
    """Given a predefined mapping, decode the sequence of numbers to a string

    Args:
        input_seq: array to decode
        mapping: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
        A string, decoded from input_seq
    """
    if not isinstance(input_seq, (Sequence, np.ndarray)):
        raise TypeError("Invalid sequence type")
    if isinstance(input_seq, np.ndarray) and (input_seq.dtype != np.int_ or input_seq.max() >= len(mapping)):
        raise AssertionError("Input must be an array of int, with max less than mapping size")

    return "".join(map(mapping.__getitem__, input_seq))


def encode_sequences(
    sequences: list[str],
    vocab: str,
    target_size: int | None = None,
    eos: int = -1,
    sos: int | None = None,
    pad: int | None = None,
    dynamic_seq_length: bool = False,
) -> np.ndarray:
    """Encode character sequences using a given vocab as mapping

    Args:
        sequences: the list of character sequences of size N
        vocab: the ordered vocab to use for encoding
        target_size: maximum length of the encoded data
        eos: encoding of End Of String
        sos: optional encoding of Start Of String
        pad: optional encoding for padding. In case of padding, all sequences are followed by 1 EOS then PAD
        dynamic_seq_length: if `target_size` is specified, uses it as upper bound and enables dynamic sequence size

    Returns:
        the padded encoded data as a tensor
    """
    if 0 <= eos < len(vocab):
        raise ValueError("argument 'eos' needs to be outside of vocab possible indices")

    if not isinstance(target_size, int) or dynamic_seq_length:
        # Maximum string length + EOS
        max_length = max(len(w) for w in sequences) + 1
        if isinstance(sos, int):
            max_length += 1
        if isinstance(pad, int):
            max_length += 1
        target_size = max_length if not isinstance(target_size, int) else min(max_length, target_size)

    # Pad all sequences
    if isinstance(pad, int):  # pad with padding symbol
        if 0 <= pad < len(vocab):
            raise ValueError("argument 'pad' needs to be outside of vocab possible indices")
        # In that case, add EOS at the end of the word before padding
        default_symbol = pad
    else:  # pad with eos symbol
        default_symbol = eos
    encoded_data: np.ndarray = np.full([len(sequences), target_size], default_symbol, dtype=np.int32)

    # Encode the strings
    for idx, seq in enumerate(map(partial(encode_string, vocab=vocab), sequences)):
        if isinstance(pad, int):  # add eos at the end of the sequence
            seq.append(eos)
        encoded_data[idx, : min(len(seq), target_size)] = seq[: min(len(seq), target_size)]

    if isinstance(sos, int):  # place sos symbol at the beginning of each sequence
        if 0 <= sos < len(vocab):
            raise ValueError("argument 'sos' needs to be outside of vocab possible indices")
        encoded_data = np.roll(encoded_data, 1)
        encoded_data[:, 0] = sos

    return encoded_data


def convert_target_to_relative(
    img: ImageTensor, target: np.ndarray | dict[str, Any]
) -> tuple[ImageTensor, dict[str, Any] | np.ndarray]:
    """Converts target to relative coordinates

    Args:
        img: tf.Tensor or torch.Tensor representing the image
        target: target to convert to relative coordinates (boxes (N, 4) or polygons (N, 4, 2))

    Returns:
        The image and the target in relative coordinates
    """
    if isinstance(target, np.ndarray):
        target = convert_to_relative_coords(target, get_img_shape(img))  # type: ignore[arg-type]
    else:
        target["boxes"] = convert_to_relative_coords(target["boxes"], get_img_shape(img))  # type: ignore[arg-type]
    return img, target


def crop_bboxes_from_image(img_path: str | Path, geoms: np.ndarray) -> list[np.ndarray]:
    """Crop a set of bounding boxes from an image

    Args:
        img_path: path to the image
        geoms: a array of polygons of shape (N, 4, 2) or of straight boxes of shape (N, 4)

    Returns:
        a list of cropped images
    """
    with Image.open(img_path) as pil_img:
        img: np.ndarray = np.array(pil_img.convert("RGB"))
    # Polygon
    if geoms.ndim == 3 and geoms.shape[1:] == (4, 2):
        return extract_rcrops(img, geoms.astype(dtype=int))
    if geoms.ndim == 2 and geoms.shape[1] == 4:
        return extract_crops(img, geoms.astype(dtype=int))
    raise ValueError("Invalid geometry format")


def pre_transform_multiclass(img, target: tuple[np.ndarray, list]) -> tuple[np.ndarray, dict[str, list]]:
    """Converts multiclass target to relative coordinates.

    Args:
        img: Image
        target: tuple of target polygons and their classes names

    Returns:
        Image and dictionary of boxes, with class names as keys
    """
    boxes = convert_to_relative_coords(target[0], get_img_shape(img))
    boxes_classes = target[1]
    boxes_dict: dict = {k: [] for k in sorted(set(boxes_classes))}
    for k, poly in zip(boxes_classes, boxes):
        boxes_dict[k].append(poly)
    boxes_dict = {k: np.stack(v, axis=0) for k, v in boxes_dict.items()}
    return img, boxes_dict
#load pretrain param
def load_pretrained_params(
    model: nn.Module,
    url: str | None = None,
    hash_prefix: str | None = None,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """Load a set of parameters onto a model

    >>> from doctr.models import load_pretrained_params
    >>> load_pretrained_params(model, "https://yoursource.com/yourcheckpoint-yourhash.zip")

    Args:
        model: the PyTorch model to be loaded
        url: URL of the zipped set of parameters
        hash_prefix: first characters of SHA256 expected hash
        ignore_keys: list of weights to be ignored from the state_dict
        **kwargs: additional arguments to be passed to `doctr.utils.data.download_from_url`
    """
    if url is None:
        logging.warning("Invalid model URL, using default initialization.")
    else:
        archive_path = download_from_url(url, hash_prefix=hash_prefix, cache_subdir="models", **kwargs)

        # Read state_dict
        state_dict = torch.load(archive_path, map_location="cpu")

        # Remove weights from the state_dict
        if ignore_keys is not None and len(ignore_keys) > 0:
            for key in ignore_keys:
                state_dict.pop(key)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if set(missing_keys) != set(ignore_keys) or len(unexpected_keys) > 0:
                raise ValueError("unable to load state_dict, due to non-matching keys.")
        else:
            # Load weights
            model.load_state_dict(state_dict)
            
#vgg16
from copy import deepcopy
from typing import Any

from torch import nn
from torchvision.models import vgg as tv_vgg

from doctr.datasets import VOCABS

from ...utils import load_pretrained_params

__all__ = ["vgg16_bn_r"]


default_cfgs: dict[str, dict[str, Any]] = {
    "vgg16_bn_r": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["vietnamese"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.4.1/vgg16_bn_r-d108c19c.pt&src=0",
    },
}


def _vgg(
    arch: str,
    pretrained: bool,
    tv_arch: str,
    num_rect_pools: int = 3,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> tv_vgg.VGG:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = tv_vgg.__dict__[tv_arch](**kwargs, weights=None)
    # list the MaxPool2d
    pool_idcs = [idx for idx, m in enumerate(model.features) if isinstance(m, nn.MaxPool2d)]
    # Replace their kernel with rectangular ones
    for idx in pool_idcs[-num_rect_pools:]:
        model.features[idx] = nn.MaxPool2d((2, 1))
    # Patch average pool & classification head
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Linear(512, kwargs["num_classes"])
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    model.cfg = _cfg

    return model


def vgg16_bn_r(pretrained: bool = False, **kwargs: Any) -> tv_vgg.VGG:
    """VGG-16 architecture as described in `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/pdf/1409.1556.pdf>`_, modified by adding batch normalization, rectangular pooling and a simpler
    classification head.

    >>> import torch
    >>> from doctr.models import vgg16_bn_r
    >>> model = vgg16_bn_r(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        **kwargs: keyword arguments of the VGG architecture

    Returns:
        VGG feature extractor
    """
    return _vgg(
        "vgg16_bn_r",
        pretrained,
        "vgg16_bn",
        3,
        ignore_keys=["classifier.weight", "classifier.bias"],
        **kwargs,
    )
#regconitionModel
# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Adapted from https://github.com/pytorch/torch/blob/master/torch/nn/modules/module.py


__all__ = ["NestedObject"]


def _addindent(s_, num_spaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class NestedObject:
    """Base class for all nested objects in doctr"""

    _children_names: list[str]

    def extra_repr(self) -> str:
        return ""

    def __repr__(self):
        # We treat the extra repr like the sub-object, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        if hasattr(self, "_children_names"):
            for key in self._children_names:
                child = getattr(self, key)
                if isinstance(child, list) and len(child) > 0:
                    child_str = ",\n".join([repr(subchild) for subchild in child])
                    if len(child) > 1:
                        child_str = _addindent(f"\n{child_str},", 2) + "\n"
                    child_str = f"[{child_str}]"
                else:
                    child_str = repr(child)
                child_str = _addindent(child_str, 2)
                child_lines.append("(" + key + "): " + child_str)
        lines = extra_lines + child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import numpy as np

from doctr.datasets import encode_sequences
from doctr.utils.repr import NestedObject

__all__ = ["RecognitionPostProcessor", "RecognitionModel"]


class RecognitionModel(NestedObject):
    """Implements abstract RecognitionModel class"""

    vocab: str
    max_length: int

    def build_target(
        self,
        gts: list[str],
    ) -> tuple[np.ndarray, list[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
            gts: list of ground-truth labels

        Returns:
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(sequences=gts, vocab=self.vocab, target_size=self.max_length, eos=len(self.vocab))
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class RecognitionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(
        self,
        vocab: str,
    ) -> None:
        self.vocab = vocab
        self._embedding = list(self.vocab) + ["<eos>"]

    def extra_repr(self) -> str:
        return f"vocab_size={len(self.vocab)}"

#main.

from collections.abc import Callable
from copy import deepcopy
from itertools import groupby
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from doctr.datasets import VOCABS, decode_sequence

from ...classification import mobilenet_v3_large_r, mobilenet_v3_small_r, vgg16_bn_r
from ...utils.pytorch import load_pretrained_params
from ..core import RecognitionModel, RecognitionPostProcessor

__all__ = ["CRNN", "crnn_vgg16_bn", "crnn_mobilenet_v3_small", "crnn_mobilenet_v3_large"]

default_cfgs: dict[str, dict[str, Any]] = {
    "crnn_vgg16_bn": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["vietnamese"],
        "url": "https://doctr-static.mindee.com/models?id=v0.3.1/crnn_vgg16_bn-9762b0b0.pt&src=0",
    },
    "crnn_mobilenet_v3_small": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["vietnamese"],
        "url": "https://doctr-static.mindee.com/models?id=v0.3.1/crnn_mobilenet_v3_small_pt-3b919a02.pt&src=0",
    },
    "crnn_mobilenet_v3_large": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["vietnamese"],
        "url": "https://doctr-static.mindee.com/models?id=v0.3.1/crnn_mobilenet_v3_large_pt-f5259ec2.pt&src=0",
    },
}


class CTCPostProcessor(RecognitionPostProcessor):
    """Postprocess raw prediction of the model (logits) to a list of words using CTC decoding

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    @staticmethod
    def ctc_best_path(
        logits: torch.Tensor,
        vocab: str = VOCABS["vietnamese"],
        blank: int = 0,
    ) -> list[tuple[str, float]]:
        """Implements best path decoding as shown by Graves (Dissertation, p63), highly inspired from
        <https://github.com/githubharald/CTCDecoder>`_.

        Args:
            logits: model output, shape: N x T x C
            vocab: vocabulary to use
            blank: index of blank label

        Returns:
            A list of tuples: (word, confidence)
        """
        # Gather the most confident characters, and assign the smallest conf among those to the sequence prob
        probs = F.softmax(logits, dim=-1).max(dim=-1).values.min(dim=1).values

        # collapse best path (using itertools.groupby), map to chars, join char list to string
        words = [
            decode_sequence([k for k, _ in groupby(seq.tolist()) if k != blank], vocab)
            for seq in torch.argmax(logits, dim=-1)
        ]

        return list(zip(words, probs.tolist()))

    def __call__(self, logits: torch.Tensor) -> list[tuple[str, float]]:
        """Performs decoding of raw output with CTC and decoding of CTC predictions
        with label_to_idx mapping dictionnary

        Args:
            logits: raw output of the model, shape (N, C + 1, seq_len)

        Returns:
            A tuple of 2 lists: a list of str (words) and a list of float (probs)

        """
        # Decode CTC
        return self.ctc_best_path(logits=logits, vocab=self.vocab, blank=len(self.vocab))


class CRNN(RecognitionModel, nn.Module):
    """Implements a CRNN architecture as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of units in the LSTM layers
        exportable: onnx exportable returns only logits
        cfg: configuration dictionary
    """

    _children_names: list[str] = ["feat_extractor", "decoder", "linear", "postprocessor"]

    def __init__(
        self,
        feature_extractor: nn.Module,
        vocab: str,
        rnn_units: int = 128,
        input_shape: tuple[int, int, int] = (3, 32, 128),
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.cfg = cfg
        self.max_length = 32
        self.exportable = exportable
        self.feat_extractor = feature_extractor

        # Resolve the input_size of the LSTM
        with torch.inference_mode():
            out_shape = self.feat_extractor(torch.zeros((1, *input_shape))).shape
        lstm_in = out_shape[1] * out_shape[2]

        self.decoder = nn.LSTM(
            input_size=lstm_in,
            hidden_size=rnn_units,
            batch_first=True,
            num_layers=2,
            bidirectional=True,
        )

        # features units = 2 * rnn_units because bidirectional layers
        self.linear = nn.Linear(in_features=2 * rnn_units, out_features=len(vocab) + 1)

        self.postprocessor = CTCPostProcessor(vocab=vocab)

        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def compute_loss(
        self,
        model_output: torch.Tensor,
        target: list[str],
    ) -> torch.Tensor:
        """Compute CTC loss for the model.

        Args:
            model_output: predicted logits of the model
            target: list of target strings

        Returns:
            The loss of the model on the batch
        """
        gt, seq_len = self.build_target(target)
        batch_len = model_output.shape[0]
        input_length = model_output.shape[1] * torch.ones(size=(batch_len,), dtype=torch.int32)
        # N x T x C -> T x N x C
        logits = model_output.permute(1, 0, 2)
        probs = F.log_softmax(logits, dim=-1)
        ctc_loss = F.ctc_loss(
            probs,
            torch.from_numpy(gt),
            input_length,
            torch.tensor(seq_len, dtype=torch.int),
            len(self.vocab),
            zero_infinity=True,
        )

        return ctc_loss

    def forward(
        self,
        x: torch.Tensor,
        target: list[str] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> dict[str, Any]:
        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        features = self.feat_extractor(x)
        # B x C x H x W --> B x C*H x W --> B x W x C*H
        c, h, w = features.shape[1], features.shape[2], features.shape[3]
        features_seq = torch.reshape(features, shape=(-1, h * c, w))
        features_seq = torch.transpose(features_seq, 1, 2)
        logits, _ = self.decoder(features_seq)
        logits = self.linear(logits)

        out: dict[str, Any] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output:
            out["out_map"] = logits

        if target is None or return_preds:
            # Disable for torch.compile compatibility
            @torch.compiler.disable  # type: ignore[attr-defined]
            def _postprocess(logits: torch.Tensor) -> list[tuple[str, float]]:
                return self.postprocessor(logits)

            # Post-process boxes
            out["preds"] = _postprocess(logits)

        if target is not None:
            out["loss"] = self.compute_loss(logits, target)

        return out


def _crnn(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[Any], nn.Module],
    pretrained_backbone: bool = True,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> CRNN:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Feature extractor
    feat_extractor = backbone_fn(pretrained=pretrained_backbone).features  # type: ignore[call-arg]

    kwargs["vocab"] = kwargs.get("vocab", default_cfgs[arch]["vocab"])
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs["vocab"]
    _cfg["input_shape"] = kwargs["input_shape"]

    # Build the model
    model = CRNN(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        load_pretrained_params(model, _cfg["url"], ignore_keys=_ignore_keys)

    return model


def crnn_vgg16_bn(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a VGG-16 backbone as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    >>> import torch
    >>> from doctr.models import crnn_vgg16_bn
    >>> model = crnn_vgg16_bn(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
        **kwargs: keyword arguments of the CRNN architecture

    Returns:
        text recognition architecture
    """
    return _crnn("crnn_vgg16_bn", pretrained, vgg16_bn_r, ignore_keys=["linear.weight", "linear.bias"], **kwargs)


def crnn_mobilenet_v3_small(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a MobileNet V3 Small backbone as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    >>> import torch
    >>> from doctr.models import crnn_mobilenet_v3_small
    >>> model = crnn_mobilenet_v3_small(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
        **kwargs: keyword arguments of the CRNN architecture

    Returns:
        text recognition architecture
    """
    return _crnn(
        "crnn_mobilenet_v3_small",
        pretrained,
        mobilenet_v3_small_r,
        ignore_keys=["linear.weight", "linear.bias"],
        **kwargs,
    )


def crnn_mobilenet_v3_large(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a MobileNet V3 Large backbone as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    >>> import torch
    >>> from doctr.models import crnn_mobilenet_v3_large
    >>> model = crnn_mobilenet_v3_large(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
        **kwargs: keyword arguments of the CRNN architecture

    Returns:
        text recognition architecture
    """
    return _crnn(
        "crnn_mobilenet_v3_large",
        pretrained,
        mobilenet_v3_large_r,
        ignore_keys=["linear.weight", "linear.bias"],
        **kwargs,
    )