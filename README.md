# DREPNet
Enhancing Small Lesion Detection in Medical Images via Dynamic Reparameterization and Edge-Polar Co-Awareness

**[Under Review at *The Visual Computer*]**

![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://link_to_your_arxiv_preprint) <!-- Optional: Add link to your arXiv preprint when available -->

This repository contains the official implementation of our proposed modules and model configurations for the paper "DREPNet: A Dynamic Reparameterized Edge-polar Co-aware Network for Small Lesion Detection".

**Note:** This is a partial release containing the core architectural components and model configuration files to ensure the reproducibility of our key designs. The full project code, including training and evaluation scripts, will be made publicly available upon the acceptance of the manuscript.

## Overview

DREPNet is a novel lightweight framework specifically engineered for the challenging task of small lesion detection in high-resolution medical images. It addresses critical issues such as low contrast, ambiguous boundaries, and information loss in conventional models by integrating a series of synergistic architectural innovations:

-   **Reparameterized Dynamic Spatial Feature Pyramid Network (RepDSFPN)**
-   **Edge-guided Multi-scale Feature Enhancement (EMFE)**
-   **Polarity-gated Attention (PGA)**

For a detailed explanation of the methodology, please refer to our manuscript.

## Core Modules and Configuration

This initial release includes the following key components to facilitate understanding and verification of our architecture:

1.  **Module Implementations (`/modules`)**:
    -   `repdsfpn_components.py`: Contains the implementation of core building blocks for RepDSFPN, such as `DySample` and `SPD-Conv`.
    -   `emfe.py`: The implementation of our Edge-guided Multi-scale Feature Enhancement (EMFE) module.
    -   `pga.py`: The implementation of our Polarity-gated Attention (PGA) mechanism.

2.  **Model Configuration (`/configs`)**:
    -   `drepnet.yaml`: The YAML configuration file that defines the full DREPNet architecture, including the backbone and head structures. This file details how our proposed modules are integrated.

## Future Release

Upon acceptance of our paper, this repository will be updated to include:
-   [ ] Complete training and validation scripts (`train.py`, `val.py`).
-   [ ] Scripts for data preprocessing.
-   [ ] Pre-trained model weights for both VinDR-CXR and Br35H datasets.
-   [ ] Detailed instructions for reproducing the results reported in our paper.

## Citation

If you find our work or the provided modules useful in your research, we kindly ask you to cite our paper. The BibTeX entry will be provided upon publication.

```bibtex
@article{YourLastName2025DREPNet,
  title={DREPNet: A Dynamic Reparameterized Edge-polar Co-aware Network for Small Lesion Detection},
  author={Li, Haoxuan and Li, Yang},
  journal={The Visual Computer},
  year={2025},
  % Details like volume, pages, publisher will be added upon acceptance.
}
```

## Dataset Information

This work was evaluated on two public datasets. To ensure originality and proper licensing, please apply for access through their official websites.

-   **VinDR-CXR**: [Official Website Link](https://vindr.ai/datasets/cxr)
-   **Br35H**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-mri-dataset)

When using these datasets, please cite their original papers accordingly.

---
