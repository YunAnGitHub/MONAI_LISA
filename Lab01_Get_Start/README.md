# Lab 01: Getting Started with MONAI

**Adapted by:** Yun-An Huang  
**Date:** May 29, 2026

This lab introduces the fundamental concepts of the **MONAI** (Medical Open Network for AI) framework. It focuses on the initial environment setup and the core mechanics of data preprocessing.

## 🎯 Objectives
The goal of this lab is to familiarize you with:
1.  **Installation**: Setting up the MONAI environment and its essential dependencies.
2.  **Array Transforms**: Learning how to apply basic transformations directly to image arrays.
3.  **Dictionary Transforms**: Utilizing the dictionary-based API (`MapTransforms`) to handle multi-modal data or image-label pairs efficiently.

## 📂 Notebook Overview
The primary script for this lab is `Lab01_Getting_Started_with_MONAI.ipynb`.

### Key Workflow:
- **Environment Setup**: Automated installation of the `monai` package and validation of the computing environment.
- **Array-Based Preprocessing**:
    - `LoadImage`: Loading medical imaging formats (e.g., NIfTI).
    - `EnsureChannelFirst`: Formatting data dimensions for PyTorch compatibility.
   
- **Dictionary-Based Preprocessing**:
    - Transitioning to the `d` suffix transforms (e.g., `LoadImaged`).
    - Managing data as dictionaries to keep images, labels, and metadata synchronized throughout the pipeline.

## 🛠 Requirements
The following libraries are utilized in this lab:
- `monai`
- `numpy`
- `nibabel` (for NIfTI file handling)
- `matplotlib` (for data visualization)

## 🚀 How to Run
1. Open `Lab01_Getting_Started_with_MONAI.ipynb` in your preferred Jupyter environment or VS Code.
2. Execute the cells sequentially to install dependencies and witness the transform effects.
3. Compare the outputs of the Array and Dictionary transform sections.
