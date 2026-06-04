# Lab 03: MONAI End-to-End Workflow

**Adapted by:** Yun-An Huang  
**Date:** June 5, 2026

This lab demonstrates the construction of a complete, reproducible medical image classification pipeline using **MONAI** with **MedNIST Dataset**. It covers everything from dataset preparation to managed training loops.

## 🎯 Objectives
The goal of this lab is to familiarize you with:
1.  **MedNIST Dataset**: Loading and processing a standard medical imaging dataset for classification.
2.  **Reproducibility**: Implementing `set_determinism` to fix random seeds and ensure consistent results during development.
3.  **Network Implementation**: Using MONAI's `DenseNet` for feature extraction and classification.
4.  **Workflow Orchestration**: Leveraging **PyTorch Ignite** integration (or via MONAI's `SupervisedTrainer` and `SupervisedEvaluator`) to automate training and validation logic.
5.  **Performance Monitoring**: Setting up event handlers to track loss, metrics, and model checkpoints.

## 📂 Notebook Overview
The primary script for this lab is `Lab03_MONAI_End_to_End_Workflow.ipynb`.

### Key Workflow:
- **Environment Configuration**: Using `set_determinism` to synchronize random states across NumPy and PyTorch.
- **Data Pipeline**: 
    - Downloading and splitting the MedNIST dataset (Hand, AbdomenCT, CXR, ChestCT, etc.).
    - Constructing efficient preprocessing transform chains.
- **Model Setup**: Initializing a `DenseNet` architecture tailored for the input image size and class count.
- **Engine-Based Training**: 
    - Utilizing the `SupervisedTrainer` for a clean, event-driven training loop.
    - Implementing the `SupervisedEvaluator` for automated validation.
- **Evaluation**: Visualizing results and calculating classification metrics.

## 🛠 Requirements
The following libraries are utilized in this lab:
- `monai`
- `torch`
- `numpy`
- `matplotlib`
- `Pillow` (for image processing)

## 🚀 How to Run
1. Open `Lab03_MONAI_End_to_End_Workflow.ipynb` in VS Code or Jupyter.
2. Run the initial cells to download the MedNIST dataset if it is not already present.
3. Execute the training pipeline and observe the Ignite engine logs for real-time performance updates.
4. Verify that the results are reproducible by re-running the notebook with the fixed seed.
