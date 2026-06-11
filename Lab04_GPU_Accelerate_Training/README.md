# Lab 04: GPU Accelerated Training

**Adapted by:** Yun-An Huang  
**Date:** June 11, 2026

This lab focuses on optimizing 3D medical image segmentation workflows using GPU acceleration and MONAI-specific performance features. It demonstrates how to transform a standard PyTorch training loop into a highly efficient pipeline and how to profile execution time using NVIDIA Nsys.

## 🎯 Objectives
The goal of this lab is to familiarize you with:
1.  **GPU Acceleration**: Leveraging CUDA for high-performance training and inference.
2.  **MONAI Fast Training**: Implementing advanced features to reduce epoch time and improve throughput.
3.  **Performance Profiling**: Using `nvtx` markers and **NVIDIA Nsys** (Nsight Systems) to visualize and analyze bottleneck stages in the pipeline.
4.  **Comparative Analysis**: Benchmarking "Regular" PyTorch training against "MONAI Optimized" training.

## 🚀 MONAI Optimization Features
This practice introduces several key features to accelerate the training process:
*   **AMP (Auto Mixed Precision)**: Utilizing `torch.cuda.amp` to significantly improve training speed on modern GPUs.
*   **CacheDataset**: Reducing I/O overhead by caching deterministic transforms in RAM.
*   **GPU-based Transforms**: Using `EnsureTyped` to move data to the GPU early, allowing random transforms to run directly on the device.
*   **ThreadDataLoader**: Using multi-threading instead of multi-processing to avoid communication overhead when data is already cached.
*   **Fast Loss Functions**: Implementing `DiceCELoss` for a robust and efficient segmentation loss.
*   **Hyperparameter Tuning**: Using `SGD` with large learning rates to exploit the stability provided by these optimizations.

## 📂 Notebook and Script Overview
This module is divided into three parts:
1.  **Core Notebook (`Lab04_GPU_Accelerate_Training.ipynb`)**: The full workflow with detailed explanations.
2.  **Profiling Notebook (`Lab04_GPU_Accelerate_Training_colab.ipynb`)**: Optimized for Google Colab, including environment setup for NVIDIA Nsys.
3.  **Profiling Script (`Lab04_GPU_Accelerate_Training_colab.py`)**: A streamlined script for automated profiling analysis with a reduced number of epochs.

## 📊 Profiling with NVIDIA Nsys
To understand where time is spent (Data Loading vs. Forward Pass vs. Optimization), we use `nvtx` annotations:
*   `nvtx.annotate("dataload")`: Tracks time spent fetching and moving data.
*   `nvtx.annotate("forward")`: Tracks model inference and loss calculation.
*   `nvtx.annotate("backward")`: Tracks gradient computation.
*   `nvtx.annotate("update")`: Tracks optimizer steps.

You can generate a profile report by running the profiling script through the `nsys` CLI to identify hardware utilization and pipeline stalls.

## 🛠 Requirements
*   `monai`
*   `torch` (with CUDA support)
*   `nvtx`
*   `matplotlib`
*   `scikit-learn` (for data splitting)
*   **NVIDIA Nsight Systems (nsys)** installed on the host machine.

