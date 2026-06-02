# Lab 02: MONAI Datasets and Caching

**Adapted by:** Yun-An Huang  
**Date:** June 2, 2026

This lab introduces the Datasets, Caching and Layer and Network modules of the **MONAI** (Medical Open Network for AI) framework. 

## 🎯 Objectives
The goal of this lab is to familiarize you with:
1.  **MONAI Datasets**: Understanding the base `Dataset` class and its compatibility with PyTorch `DataLoader`.
2.  **Caching Mechanisms**: Learning how to accelerate training using `CacheDataset`, `PersistentDataset`, and `SmartCacheDataset`.
3.  **Public Datasets**: Utilizing `monai.apps` to download and load standard datasets like the Medical Segmentation Decathlon (MSD).
4.  **Network Components**: Using MONAI layer factories (`Conv`, `Act`, `Pool`) to build flexible, dimension-agnostic architectures.
5.  **Pre-built Networks**: Instantiating and customizing complex architectures such as `UNet` and `DenseNet`.

## 📂 Notebook Overview
The primary script for this lab is `Lab02_MONAI_Datasets_and_Caching.ipynb`.

### Key Workflow:
- **Dataset Fundamentals**: Creating basic datasets and applying custom transforms (e.g., `SquareIt`).
- **Performance Optimization**: 
    - **CacheDataset**: Storing transforms in RAM for rapid access.
    - **PersistentDataset**: Caching non-random transforms on disk to save memory.
    - **SmartCacheDataset**: Dynamically replacing a subset of the cache to handle large datasets.
- **Decathlon Dataset**: Hands-on experience with the Spleen segmentation dataset.
- **Network Design**:
    - Exploring `monai.networks.layers` for generic layer creation.
    - Building a custom `MyNetwork` that works for both 2D and 3D.
    - Deep dive into `UNet` configurations and overriding internal methods to use `ResidualUnits`.

## 🛠 Requirements
The following libraries are utilized in this lab:
- `monai`
- `torch`
- `numpy`
- `matplotlib`

## 🚀 How to Run
1. Open `Lab02_MONAI_Datasets_and_Caching.ipynb` in VS Code or Jupyter.
2. Execute the cells sequentially to install dependencies and witness the transform effects.
3. Observe the time differences between standard and cached data loading.
