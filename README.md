# Enhancing Small Lesion Detection in Medical Images via Dynamic Reparameterization and Edge-Polar Co-Awareness

**[Manuscript Submitted to *The Visual Computer*]**

This repository contains the official implementation of the core architectural components for our paper, "Enhancing Small Lesion Detection in Medical Images via Dynamic Reparameterization and Edge-Polar Co-Awareness". The proposed model is named **DREPNet**.

**Note on Code Availability:** This is a partial release intended to ensure the transparency and reproducibility of our key architectural designs. The full project, including scripts for training, inference, and visualization, along with dependency files, will be made publicly available shortly after the manuscript's acceptance.

## Overview

This work introduces **DREPNet**, a novel lightweight framework specifically engineered for the challenging task of small lesion detection in high-resolution medical images. To address critical issues such as low contrast, ambiguous boundaries, and information loss in conventional models, DREPNet integrates a series of synergistic architectural innovations. These include a high-fidelity reparameterized feature pyramid for lossless feature scaling, a specialized multi-scale mechanism for edge enhancement, and a polarity-aware attention module for improved feature discriminability. Our goal is to provide an efficient, reliable, and practical solution for computer-aided early diagnosis.

## Current Release

This repository currently contains the core components of the DREPNet architecture:

1.  **Module Implementations (`/modules`)**:
    -   This directory contains the PyTorch implementation of the core modules for the DREPNet model.

2.  **Model Configuration (`/config`)**:
    -   The `DREPNet.yaml` file, which defines the complete model architecture and demonstrates how the modules are integrated.

## Datasets

Our model's performance was rigorously evaluated on two challenging, publicly available medical imaging datasets. We encourage users to access these datasets via their official sources to ensure compliance with their respective licensing and usage policies.

-   **VinDR-CXR** ([Official Website](https://vindr.ai/datasets))

    The VinDR-CXR dataset is a large-scale collection of 18,000 chest X-ray images, annotated by experienced radiologists for 14 common thoracic diseases. In our study, we utilized the 15,000 images with public annotations. This dataset is particularly challenging due to the high prevalence of small and low-contrast lesions.

-   **Br35H** ([Official Website](https://ieee-dataport.org/documents/br35h-brain-tumor-detection-2020-0))

    The Br35H dataset contains 801 brain MRI images labeled for tumor detection. For our experiments, we followed a standard split of 500 images for training, 201 for validation, and 100 for testing. This dataset serves as a benchmark for evaluating the model's generalization ability on a different imaging modality and anatomical region.

Please ensure to cite the original papers of these datasets if you use them in your research.
