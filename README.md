---
title: Galaxy_Morphology_Classification_using_CNNs_and_ViT
app_file: inference/app.py
sdk: gradio
sdk_version: 6.2.0
---
# Galaxy Morphology Classification (PyTorch)
---

## Overview

This repository contains a clean, reproducible, and research oriented implementation of galaxy morphology classification using PyTorch.

The project re implements and extends prior research work on galaxy morphology classification, originally conducted using TensorFlow, and transitions it into a PyTorch first, engineering grade pipeline. The focus is on correctness, reproducibility, scalability, and clean dataset handling rather than quick experimentation.

The project uses the `Galaxy10 DECaLS` dataset as the initial benchmark and is designed to be easily extensible to larger datasets such as Galaxy Zoo and DECaLS full releases.

<br>

## Motivation

Galaxy morphology classification is a fundamental task in astrophysics, providing insight into galaxy formation and evolution. While many prior works rely on small in memory datasets and framework specific loaders, this project emphasizes:

- Framework agnostic dataset handling
- Disk backed datasets compatible with large scale training
- Clean PyTorch data pipelines
- Research reproducibility and clarity
- Industry level project structure

The goal is to build a foundation suitable for:

- Research extensions
- Large scale experiments
- Open source collaboration
- Future deployment and demos

<br>

## Dataset

### Galaxy10 DECaLS

The `Galaxy10 DECaLS` dataset consists of 17,736 RGB galaxy images of resolution 256 × 256, categorized into 10 morphological classes.

The original dataset is distributed as a single HDF5 file via the astroNN library. In this project, the dataset is exported once into an ImageFolder compatible directory structure to enable PyTorch native workflows.

**Class Labels**

The dataset uses the following class mapping:

| Label | Class Name            |
| ----- | --------------------- |
| 0     | Disturbed             |
| 1     | Merging               |
| 2     | Round Smooth          |
| 3     | Smooth, Cigar Shaped  |
| 4     | Cigar Shaped Smooth   |
| 5     | Barred Spiral         |
| 6     | Unbarred Tight Spiral |
| 7     | Unbarred Loose Spiral |
| 8     | Edge on without Bulge |
| 9     | Edge on with Bulge    |

*This class ordering matches the original Galaxy10 DECaLS specification.*

<br>

## Dataset Preparation Pipeline

The dataset preparation follows a one time extraction workflow:

1. Galaxy10 DECaLS is downloaded automatically via `astroNN`
2. The raw `.h5` file is cached outside the repository
3. Images are exported into a disk based directory structure
4. Data is split into train, validation, and test sets
5. PyTorch `ImageFolder` is used for all experiments

After export, `astroNN` is no longer required.


**Final Dataset Structure**
```
data/Galaxy10_DECaLS/
├── train/
│   ├── barred_spiral/
│   ├── merging/
│   └── ...
├── val/
│   ├── barred_spiral/
│   └── ...
└── test/
    ├── barred_spiral/
    └── ...
``` 
*The `data/` directory is git ignored and treated as immutable once generated.*

<br>

## Project Structure

``` 
galaxy_morphology_classification/
├── data/                    # git ignored, processed datasets
│   └── Galaxy10_DECaLS/
├── datasets/                # PyTorch dataset utilities
├── models/                  # CNN and Transformer models
├── scripts/
│   └── export_galaxy10.py   # one time dataset export script
├── training/                # training loops and trainers
├── utils/                   # helpers, metrics, logging
├── .gitignore
└── README.md
```
*This structure is intentionally modular and scalable.*

