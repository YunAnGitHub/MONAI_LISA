#!/usr/bin/env python
# coding: utf-8


# **Adapted by:** Yun-An Huang  
# **Date:** June 8th, 2026
# 
# This practice module is divided into three parts:
# 1. Core Notebook (Lab04_GPU_Accelerate_Training.ipynb): A comprehensive Jupyter Notebook containing detailed explanations and the complete workflow for this exercise.
# 2. Profiling Notebook (Lab04_GPU_Accelerate_Training_colab.ipynb): A Jupyter Notebook designed to run on Google Colab or a local server for performance profiling. This part includes the complete environment setup, including NVIDIA Nsys configuration.
# 3. Profiling Script (Lab04_GPU_Accelerate_Training_colab.py): A streamlined Python script dedicated strictly to profiling analysis. It omits the environment setup and utilizes a reduced number of training epochs for quick execution. This script is intended to be run after completing the setup in the profiling notebook.

# please run the Profiling Notebook before this notebook.

## Initialization
from google.colab import drive # Import Google Colab drive utility # type: ignore
import os # Import os module for interacting with the operating system

import monai # type: ignore
import torch
import glob
import math
import os
import shutil
import tempfile
import time

import matplotlib.pyplot as plt

from torch.optim import Adam, SGD
from monai.apps import download_and_extract # type: ignore
from monai.config import print_config # type: ignore
from monai.data import ( # type: ignore
    CacheDataset,
    DataLoader,
    ThreadDataLoader,
    Dataset,
    decollate_batch,
    set_track_meta,
) 
from monai.inferers import sliding_window_inference # type: ignore
from monai.losses import DiceLoss, DiceCELoss # type: ignore
from monai.metrics import DiceMetric # type: ignore
from monai.networks.layers import Act, Norm # type: ignore
from monai.networks.nets import UNet # type: ignore
from monai.transforms import ( # type: ignore
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureTyped,
    FgBgToIndicesd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import set_determinism # type: ignore

# for profiling
import nvtx # type: ignore
from monai.utils.nvtx import Range # type: ignore
import contextlib  # to improve code readability (combining training/validation loop with and without profiling)

print_config()

print("GPU Avalable:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU type:", torch.cuda.get_device_name(0))

# Make sure you have been linked to g-drive. otherwise please comment this line:
PROJECT_SANDBOX = "/content/drive/MyDrive/Colab_Workspace/MONAI_LISA" # Define the project sandbox directory within Google Drive

# ## Setup data & output directories
# 
directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory
print(f"root dir is: {root_dir}")

out_dir = os.path.join(PROJECT_SANDBOX,"Lab04","outputs/")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print(out_dir)


# ## Profiling
profiling = True

# to see the trend in training curve and dice results, set max_epochs to be larger (300)
# note that before optimization, training can be quite a bit slower
if profiling:
    max_epochs = 6
else:
    max_epochs = 300

# to improve readability
def range_func(x, y):
    return Range(x)(y) if profiling else y

no_profiling = contextlib.nullcontext() # no_profiling is not a boolean value, it is a null context manager.

# ## Download dataset
resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
data_root = os.path.join(root_dir, "Task09_Spleen")
if not os.path.exists(data_root):
    download_and_extract(resource, compressed_file, root_dir, md5)


# ## Set MSD Spleen dataset path
from sklearn.model_selection import train_test_split

train_images = sorted(glob.glob(os.path.join(data_root, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_root, "labelsTr", "*.nii.gz")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

# the script do not use a random split, it is recommended to use a random split.
#train_files, val_files = data_dicts[:-9], data_dicts[-9:]

# random split, use seed = 0 for debugging.
train_files, val_files = train_test_split(data_dicts, test_size=9, random_state=0)


# ## Setup transforms for training and validation
def transformations(fast=False, device="cuda:0"):
    train_transforms = [
        range_func("LoadImage", LoadImaged(keys=["image", "label"])),
        range_func("EnsureChannelFirst", EnsureChannelFirstd(keys=["image", "label"])),
        range_func("Orientation", Orientationd(keys=["image", "label"], axcodes="RAS")),
        range_func(
            "Spacing",
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),

        ),
        range_func( # the Range of CT(HU value) for body tissues.
            "ScaleIntensityRange", 
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ),
        range_func("CropForeground", CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True)),
        # pre-compute foreground and background indexes
        # and cache them to accelerate training
        range_func(
            "Indexing",
            FgBgToIndicesd(
                keys="label",
                fg_postfix="_fg",
                bg_postfix="_bg",
                image_key="image",
            ),
        ),
    ]

    if fast:
        # convert the data to Tensor without meta, move to GPU and cache to avoid CPU -> GPU sync in every epoch
        train_transforms.append(
            range_func("EnsureType", EnsureTyped(keys=["image", "label"], device=device, track_meta=False))
        )

    train_transforms.append(
        # randomly crop out patch samples from big
        # image based on pos / neg ratio
        # the image centers of negative samples
        # must be in valid image area
        range_func(
            "RandCrop",
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                fg_indices_key="label_fg",
                bg_indices_key="label_bg",
            ),
        ),
    )

    val_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
    ]
    if fast:
        # convert the data to Tensor without meta, move to GPU and cache to avoid CPU -> GPU sync in every epoch
        val_transforms.append(EnsureTyped(keys=["image", "label"], device=device, track_meta=False))

    return Compose(train_transforms), Compose(val_transforms)


# ## Define the training progress
# For a typical PyTorch regular training procedure, use regular `Dataset`, `DataLoader`, `Adam` optimizer and `Dice` loss to train the model.
# 
# For MONAI fast training progress, we mainly introduce the following features:
# 1. `AMP` (auto mixed precision): AMP is an important feature released in PyTorch v1.6, NVIDIA CUDA 11 added strong support for AMP and significantly improved training speed.
# 2. `CacheDataset`: Dataset with the cache mechanism that can load data and cache deterministic transforms' result during training.
# 3. `EnsureTyped` transform: to move data to GPU and cache with `CacheDataset`, then execute random transforms on GPU directly, avoid CPU -> GPU sync in every epoch. Please note that not all the MONAI transforms support GPU operation so far, still working in progress.
# 4. `set_track_meta(False)`: to disable meta tracking in the random transforms to avoid unnecessary computation.
# 5. `ThreadDataLoader`: uses multi-threads instead of multi-processing, faster than `DataLoader` in light-weight task as we already cached the results of most computation.
# 6. `DiceCE` loss function: computes Dice loss and Cross Entropy Loss, returns the weighted sum of these two losses.
# 7. Analyzed the training curve and tuned algorithm: Use `SGD` optimizer, different network parameters, etc.
# 
# (A note on code: to improve readability and support the profiling flag, we used the `with nvtx(...) if profiling else no_profiling` context pattern, where `no_profiling` is a null context from Python's native `contextlib` with no effect on the code. An acknowledgement is provided here[<sup id="fn1-back">1</sup>](#fn1).)

def train_process(fast=False):
    learning_rate = 2e-4
    val_interval = 5  # do validation for every epoch
    set_track_meta(True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise RuntimeError("this tutorial is intended for GPU, but no CUDA device is available")

    train_trans, val_trans = transformations(fast=fast, device=device)
    # set CacheDataset, ThreadDataLoader and DiceCE loss for MONAI fast training
    if fast:
        # as `RandCropByPosNegLabeld` crops from the cached content and `deepcopy`
        # the crop area instead of modifying the cached value, we can set `copy_cache=False`
        # to avoid unnecessary deepcopy of cached content in `CacheDataset`
        train_ds = CacheDataset(
            data=train_files,
            transform=train_trans,
            cache_rate=1.0,
            num_workers=8,
            copy_cache=False,
        )
        val_ds = CacheDataset(data=val_files, transform=val_trans, cache_rate=1.0, num_workers=5, copy_cache=False)
        # disable multi-workers because `ThreadDataLoader` works with multi-threads
        train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=4, shuffle=True)
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

        loss_function = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            batch=True,
            smooth_nr=0.00001,
            smooth_dr=0.00001,
            lambda_dice=0.5,
            lambda_ce=0.5,
        )
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            kernel_size=3,
            up_kernel_size=3,
            act=Act.PRELU,
            dropout=0.2,
            bias=True,
        ).to(device)
        # avoid the computation of meta information in random transforms
        set_track_meta(False)
    else:
        train_ds = Dataset(data=train_files, transform=train_trans)
        val_ds = Dataset(data=val_files, transform=val_trans)
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    if fast:
        # SGD prefer to much bigger learning rate
        optimizer = SGD(
            model.parameters(),
            lr=learning_rate * 1000,
            momentum=0.9,
            weight_decay=0.00004,
        )
        scaler = torch.GradScaler("cuda")
    else:
        optimizer = Adam(model.parameters(), learning_rate)

    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    epoch_times = []
    total_start = time.time()

    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")

        # profiling: full epoch
        with nvtx.annotate("epoch", color="red") if profiling else no_profiling:
            model.train()
            epoch_loss = 0
            train_loader_iterator = iter(train_loader) # avoid using for loop to avoid memory cleaning after for loop.

            # using step instead of iterate through train_loader directly to track data loading time
            # steps are 1-indexed for printing and calculation purposes
            for step in range(1, len(train_loader) + 1):
                step_start = time.time()

                # profiling: train dataload
                with nvtx.annotate("dataload", color="red") if profiling else no_profiling:
                    # rng_train_dataload = nvtx.start_range(message="dataload", color="red")
                    batch_data = next(train_loader_iterator)
                    inputs, labels = (
                        batch_data["image"].to(device),
                        batch_data["label"].to(device),
                    )

                optimizer.zero_grad()
                # set AMP for MONAI training
                if fast:
                    # profiling: forward
                    with nvtx.annotate("forward", color="green") if profiling else no_profiling:
                        with torch.autocast("cuda"):
                            outputs = model(inputs)
                            loss = loss_function(outputs, labels)

                    # profiling: backward
                    with nvtx.annotate("backward", color="blue") if profiling else no_profiling:
                        scaler.scale(loss).backward()

                    # profiling: update
                    with nvtx.annotate("update", color="yellow") if profiling else no_profiling:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    # profiling: forward
                    with nvtx.annotate("forward", color="green") if profiling else no_profiling:
                        outputs = model(inputs)
                        loss = loss_function(outputs, labels)

                    # profiling: backward
                    with nvtx.annotate("backward", color="blue") if profiling else no_profiling:
                        loss.backward()

                    # profiling: update
                    with nvtx.annotate("update", color="yellow") if profiling else no_profiling:
                        optimizer.step()

                epoch_loss += loss.item()
                epoch_len = math.ceil(len(train_ds) / train_loader.batch_size)
                print(
                    f"{step}/{epoch_len}, train_loss: {loss.item():.4f}" f" step time: {(time.time() - step_start):.4f}"
                )
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loader_iterator = iter(val_loader)

                    for _ in range(len(val_loader)):
                        # profiling: val dataload
                        with nvtx.annotate("dataload", color="red") if profiling else no_profiling:
                            val_data = next(val_loader_iterator)
                            val_inputs, val_labels = (
                                val_data["image"].to(device),
                                val_data["label"].to(device),
                            )

                        roi_size = (160, 160, 160)
                        sw_batch_size = 4

                        # profiling: sliding window
                        with nvtx.annotate("sliding window", color="green") if profiling else no_profiling:
                            # set AMP for MONAI validation
                            if fast:
                                with torch.autocast("cuda"):
                                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                            else:
                                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

                        # profiling: decollate batch
                        with nvtx.annotate("decollate batch", color="blue") if profiling else no_profiling:
                            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                            val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                        # profiling: compute metric
                        with nvtx.annotate("compute metric", color="yellow") if profiling else no_profiling:
                            dice_metric(y_pred=val_outputs, y=val_labels)

                    metric = dice_metric.aggregate().item()
                    dice_metric.reset()
                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        best_metrics_epochs_and_time[0].append(best_metric)
                        best_metrics_epochs_and_time[1].append(best_metric_epoch)
                        best_metrics_epochs_and_time[2].append(time.time() - total_start)
                        torch.save(model.state_dict(), os.path.join(out_dir, "best_metric_model.pt"))
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current"
                        f" mean dice: {metric:.4f}"
                        f" best mean dice: {best_metric:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )
        print(f"time consuming of epoch {epoch + 1} is:" f" {(time.time() - epoch_start):.4f}")
        epoch_times.append(time.time() - epoch_start)

    total_time = time.time() - total_start
    print(
        f"train completed, best_metric: {best_metric:.4f}"
        f" at epoch: {best_metric_epoch}"
        f" total time: {total_time:.4f}"
    )
    return (
        max_epochs,
        epoch_loss_values,
        metric_values,
        epoch_times,
        best_metrics_epochs_and_time,
        total_time,
    )


# ## Enable determinism and execute regular PyTorch training

set_determinism(seed=0)
regular_start = time.time()
(
    epoch_num,
    epoch_loss_values,
    metric_values,
    epoch_times,
    best,
    train_time,
) = train_process(fast=False)
total_time = time.time() - regular_start
print(f"total time of {epoch_num} epochs with regular PyTorch training: {total_time:.4f}")

# ## Enable determinism and execute MONAI optimized training
set_determinism(seed=0)
monai_start = time.time()
(
    epoch_num,
    m_epoch_loss_values,
    m_metric_values,
    m_epoch_times,
    m_best,
    m_train_time,
) = train_process(fast=True)
m_total_time = time.time() - monai_start
print(
    f"total time of {epoch_num} epochs with MONAI fast training: {m_train_time:.4f},"
    f" time of preparing cache: {(m_total_time - m_train_time):.4f}"
)
# ## Cleanup data directory
# 
# Remove directory if a temporary was used.
if directory is None:
    shutil.rmtree(root_dir)
