import os
import tarfile
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import torch
import torch.optim as optim
# from torchsummaryX import summary

from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from mltu.torch.metrics import CERMetric, WERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, ReduceLROnPlateau, WandbCallback

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from model import Network
from configs import ModelConfigs


configs = ModelConfigs()
# Save vocab and maximum text length to configs
configs.vocab = "".join(vocab)
configs.max_text_length = max_len

# configs.height = 32
# configs.width = 128
configs.height = 32 #32
configs.width = 128 #128
configs.batch_size = 64
configs.train_epochs = 5
# configs.learning_rate=0.0001
configs.save()
print(len(configs.vocab))
max_samples = 120000
limited_dataset = dataset[:max_samples]

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        # ImageShowCV2(), # uncomment to show images when iterating over the data provider
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
    ],
#     use_cache=True,
)

# Clear the original dataset to save memory
#del dataset

# Split the dataset into training and validation sets
train_dataProvider, val_dataProvider = data_provider.split(split = 0.8)
val_dataProvider, test_dataProvider = val_dataProvider.split(split = 0.6)

# Augment training data with random brightness, rotation, and erode/dilate
train_dataProvider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10), 
    ]

network = Network(len(configs.vocab))
loss = CTCLoss(blank=len(configs.vocab))
optimizer = optim.Adam(network.parameters(), lr=configs.learning_rate)

# uncomment to print network summary, torchsummaryX package is required
# summary(network, torch.zeros((1, configs.height, configs.width, 3)))

# put on cuda device if available
# if torch.cuda.is_available():
#     network = network.cuda()
network = network.to(device)
# create callbacks
wandb_callback = WandbCallback(
    project_name="HTR_Recognition",
    run_name="experiment_1",
    config={
        "learning_rate": configs.learning_rate,
        "batch_size": configs.batch_size,
        "vocab_size": len(configs.vocab),
        "max_text_length": configs.max_text_length,
        # Add any other hyperparameters you want to track
    }
)

earlyStopping = EarlyStopping(monitor="val_CER", patience=20, mode="min", verbose=1)
modelCheckpoint = ModelCheckpoint(configs.model_path + "/model.pt", monitor="val_CER", mode="min", save_best_only=True, verbose=1)
tb_callback = TensorBoard(configs.model_path + "/logs")
reduce_lr = ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=10, verbose=1, mode="min", min_lr=1e-6)
model2onnx = Model2onnx(
    saved_model_path=configs.model_path + "/model.pt",
    input_shape=(1, configs.height, configs.width, 3), 
    verbose=1,
    metadata={"vocab": configs.vocab}
    )

# create model object that will handle training and testing of the network
model = Model(network, optimizer, loss, metrics=[CERMetric(configs.vocab), WERMetric(configs.vocab)])
model.fit(
    train_dataProvider, 
    val_dataProvider, 
    epochs=configs.train_epochs, 
    callbacks=[earlyStopping, modelCheckpoint, tb_callback, reduce_lr, model2onnx, wandb_callback]
    )
print("finish")

# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))
test_dataProvider.to_csv(os.path.join(configs.model_path, "test.csv"))
