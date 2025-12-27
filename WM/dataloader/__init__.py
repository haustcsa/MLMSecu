import os
from .dataset import attrsImgDataset
from config import training_config as cfg
from torch.utils.data import DataLoader

train_dataset = attrsImgDataset(
    os.path.join(cfg.dataset_path, "train_" + str(cfg.image_size)),
    cfg.image_size,
    "celebahq",
)

val_dataset = attrsImgDataset(
    os.path.join(cfg.dataset_path, "val_" + str(cfg.image_size)),
    cfg.image_size,
    "celebahq",
)

test_dataset = attrsImgDataset(
    os.path.join(cfg.dataset_path, "test_" + str(cfg.image_size)),
    cfg.image_size,
    "celebahq",
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False
)


