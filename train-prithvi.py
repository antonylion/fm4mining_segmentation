import os
import torch
import warnings
from segmentation_models_pytorch.encoders import encoders as smp_encoders
import rasterio
import numpy as np
from terratorch.models import PrithviModelFactory
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2  # Optional if you want to convert images to tensors directly

DATASET_PATH = '/dss/dsstbyfs02/pn49cu/pn49cu-dss-0016/terratorch_fasteo/GhanaMining'

ghana_mining_bands = [
    "BLUE",
    "GREEN",
    "RED",
    "VNIR_1",
    "VNIR_2",
    "VNIR_3",
    "VNIR_4",
    "VNIR_5",
    "SWIR_1",
    "SWIR_2",
    "VV",
    "VH",
    "DEM"
]

#MEANS AND STDS FOR EACH BAND
means=[
        1473.81388377,
        1703.35249650,
        1696.67685941,
        2105.88077538,
        3133.47063806,
        3543.17834759,
        3677.26573009,
        3832.39764247,
        3156.11122121,
        2226.06822112,
        -6.52798491,
        -3.60578219,
        192.83874697,
    ]
stds=[
        223.43533204,
        285.53613398,
        413.82320306,
        395.83356682,
        312.48236897,
        354.47004513,
        397.92163014,
        389.61483882,
        451.49534791,
        468.26765909,
        6.66565737,
        3.96725815,
        57.37162747,
    ]

train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ])

datamodule = GenericNonGeoSegmentationDataModule(
    batch_size=4,
    num_workers=79,
    train_data_root=os.path.join(DATASET_PATH, 'training'),
    val_data_root=os.path.join(DATASET_PATH, 'validation'),
    test_data_root=os.path.join(DATASET_PATH, 'validation'), # We are reusing the validation set for testing 
    img_grep="*_IMG.tif",
    label_grep="*_MASK.tif",
    means=means,
    stds=stds,
    num_classes=2,

    # if transforms are defined with Albumentations, you can pass them here
    train_transform=train_transform,
    #val_transform=val_transform,
    #test_transform=val_transform,

    # Bands of your dataset (in this case similar to the model bands)
    dataset_bands=ghana_mining_bands,
    # Input bands of your model
    output_bands=ghana_mining_bands,
    no_data_replace=0,
    no_label_replace=-1,
)

model_args = {
        "backbone":"prithvi_vit_100", # see smp_encoders.keys()
        'model': 'UperNetDecoder', # 'DeepLabV3', 'DeepLabV3Plus', 'FPN', 'Linknet', 'MAnet', 'PAN', 'PSPNet', 'Unet', 'UnetPlusPlus' 
        "bands": ghana_mining_bands,
        "in_channels": 13,
        "num_classes": 2,
        "pretrained": True,
}

task = SemanticSegmentationTask(
    model_args=model_args,
    model_factory="PrithviModelFactory",
    loss="ce",
    lr=1e-4,
    ignore_index=-1,
    optimizer="AdamW",
    optimizer_hparams={"weight_decay": 0.05},
    freeze_backbone=True,
    class_names=['Non_mining', 'Mining'],
    class_weights=[0.1, 0.9]
)

datamodule.setup("fit")
checkpoint_callback = ModelCheckpoint(monitor=task.monitor, save_top_k=1, save_last=True)
early_stopping_callback = EarlyStopping(monitor=task.monitor, min_delta=0.00, patience=20)
logger = TensorBoardLogger(save_dir='output', name='resnet50')

trainer = Trainer(
    devices=4, # Number of GPUs. Interactive mode recommended with 1 device
    precision="16-mixed",
    callbacks=[
        RichProgressBar(),
        checkpoint_callback,
        #early_stopping_callback,
        #LearningRateMonitor(logging_interval="epoch"),
    ],
    logger=logger,
    max_epochs=100,
    default_root_dir='output/resnet50',
    log_every_n_steps=1,
    check_val_every_n_epoch=1
)
_ = trainer.fit(model=task, datamodule=datamodule)

# mIOU is Multiclass_Jaccard_Index
res = trainer.test(model=task, datamodule=datamodule)