# Galaxy Morphology Classification (PyTorch)
---

## Live Demo

🔗 Hugging Face Space:  
https://huggingface.co/spaces/Rishabh-9090/Galaxy_Morphology_Classification_using_CNNs_and_ViT


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
data/
├── Galaxy10_DECaLS/
│   ├── train/
│   │   ├── barred_spiral/
│   │   ├── merging/
│   │   └── ...
│   ├── val/
│   │   ├── barred_spiral/
│   │   └── ...
│   └── test/
│       ├── barred_spiral/
│       └── ...
│
├── Galaxy10_DECaLS_Balanced/
│   ├── train/
│   │   ├── barred_spiral/
│   │   ├── merging/
│   │   └── ...
│   └── test/
│       ├── barred_spiral/
│       └── ...


``` 
*The `data/` directory is git ignored and treated as immutable once generated.*

<br>

## Project Structure

``` 
galaxy_morphology_classification/
├── checkpoints/                     # (git-ignored) trained model weights
│   ├── custom_cnn/
│   ├── resnet18/
│   ├── resnet26/
│   ├── resnet50/
│   ├── vgg16/
│   ├── vgg19/
│   └── vit/
│
├── data/                            # (git-ignored) processed datasets
│   ├── Galaxy10_DECaLS/             # original class-imbalanced dataset
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── Galaxy10_DECaLS_Balanced/    # class-balanced variant
│       ├── train/
│       └── test/
│
├── inference/
│   ├── app.py                      # Gradio + Hugging Face Spaces app
│   └── config.py                   # inference-specific configuration
│
├── notebooks/
│   ├── eda.ipynb                   # exploratory data analysis
│   └── train_and_evaluate.ipynb    # experimentation & ablation studies
│
├── results/                        # evaluation outputs (metrics, plots)
│
├── src/                             # core library code
│   ├── data/
│   │   ├── dataloaders.py
│   │   └── transforms.py
│   │
│   ├── models/
│   │   ├── custom_cnn.py
│   │   ├── resnet.py
│   │   ├── vgg.py
│   │   ├── vit.py
│   │   └── get_model.py
│   │
│   ├── training/
│   │   └── engine.py               # unified training loop
│   │
│   ├── evaluation/
│   │   └── predictions.py
│   │
│   └── utils/
│       ├── early_stopping.py
│       ├── save_model.py
│       ├── set_seed.py
│       └── visualisations.py
│
├── pyproject.toml                  # packaging & tooling metadata
├── requirements.txt                # runtime dependencies
├── LICENSE
├── .gitignore
└── README.md
```
<br>

## Experimental Results

This section summarizes the test set performance of all evaluated models under different training and evaluation conditions.
Two training regimes were considered:

- Large unbalanced dataset (original class distribution)

- Balanced dataset (class balanced via resampling)

Each balanced model was evaluated on both:

- A balanced test set

- The original unbalanced test set

All metrics reported below correspond to held out test data.

### CNN and Transformer Performance Overview

| Model| Training Data | Test Data|Test Accuracy|Test Loss|
|---|---|---|---|---|
| ResNet50|Unbalanced|Unbalanced|0.5259|1.3268|
| ResNet50|Balanced|Balanced|0.4833|1.4588|
| ResNet50|Balanced|Unbalanced|0.4752|1.4446|
| ResNet26|Unbalanced|Unbalanced|0.5169|1.3493|
| ResNet26|Balanced|Balanced|0.4924|1.4766|
| ResNet26|Balanced|Unbalanced|0.4899|1.4552|
| ResNet18|Unbalanced|Unbalanced|0.4707|1.4653|
| ResNet18|Balanced|Balanced|0.4667|1.5569|
| ResNet18|Balanced|Unbalanced|0.4504|1.5166|
| VGG16|Unbalanced|Unbalanced|0.6071|1.1234|
| VGG16|Balanced|Balanced|0.5394|1.2986|
| VGG16|Balanced|Unbalanced|0.5349|1.3320|
| VGG19|Unbalanced|Unbalanced|0.5688|1.2092|
| VGG19|Balanced|Balanced|0.5106|1.3947|
| VGG19|Balanced|Unbalanced|0.5118|1.3717|
| Custom CNN|Unbalanced|Unbalanced|0.7514|0.8144|
| Custom CNN|Balanced|Balanced|0.5924|1.1362|
| Custom CNN|Balanced|Unbalanced|0.6240|1.0804|
| ViT|Unbalanced|Unbalanced|0.7638|0.7343|
| ViT|Balanced|Balanced|0.7500|0.7966|
| ViT|Balanced|Unbalanced|0.7694|0.7158|

### Key Observations

1. **Vision Transformer achieves the best overall performance**

The Vision Transformer consistently outperforms all CNN based architectures across both balanced and unbalanced evaluation settings. Its performance remains stable even when trained on balanced data and evaluated on the original unbalanced distribution, indicating strong generalization.

This suggests that global self attention is particularly effective for galaxy morphology classification, especially for complex and irregular structures.

2. **Custom CNN performs competitively with ViT**

The custom CNN achieves strong performance, closely approaching the Vision Transformer on the unbalanced dataset. This indicates that with appropriate architectural design, convolutional models can still be highly effective for this task.

However, the drop in performance when trained on balanced data highlights the sensitivity of CNNs to changes in data distribution.

3. **VGG models outperform ResNet variants**

Among CNN based architectures, VGG16 and VGG19 consistently outperform ResNet18, ResNet26, and ResNet50. In particular, VGG19 trained on balanced data demonstrates improved generalization across both balanced and unbalanced test sets.

This may be attributed to the deeper sequential convolutional structure of VGG models, which appears better suited to capturing fine grained morphological features.

4. **ResNet depth does not correlate with better performance**

Increasing ResNet depth does not lead to improved accuracy in this task. ResNet50 performs only marginally better than ResNet26 and ResNet18, and in some cases performs worse when trained on balanced data.

This suggests that residual depth alone is insufficient for modeling the structural complexity present in galaxy morphology images.

5. **Consistent failure of CNNs on disturbed galaxies**

Across all convolutional architectures evaluated, the disturbed galaxy class is consistently the most poorly classified. This behavior persists regardless of model depth, architecture type, or dataset balancing strategy.

In contrast, the Vision Transformer does not exhibit the same degradation on this class, indicating a stronger ability to model global irregularities, asymmetries, and long range spatial dependencies that characterize disturbed morphologies.

6. **Impact of dataset balancing**

Training on a balanced dataset generally improves class level fairness but often leads to reduced overall accuracy on the original unbalanced distribution. This trade off is particularly evident in CNN based models.

The Vision Transformer is notably less affected by this trade off, maintaining high performance across both evaluation settings.

### Summary

Overall, the results demonstrate that:

- *Vision Transformers provide the most robust and generalizable performance for galaxy morphology classification*

- *Carefully designed CNNs can still achieve competitive results*

- *Dataset balancing introduces important trade offs that must be evaluated in context*

- *Disturbed and irregular morphologies remain challenging for convolutional architectures*

- *These findings support the use of transformer based models for future large scale galaxy morphology studies.*
<br>

## Future Work

Planned extensions include:

- Vision Transformers provide the most robust and generalizable performance for galaxy morphology classification

- Carefully designed CNNs can still achieve competitive results

- VGG style architectures outperform residual networks in this task

- Dataset balancing introduces important trade offs that must be evaluated in context

- Disturbed and highly irregular morphologies remain challenging for convolutional architectures

- Transformer based models offer a clear advantage for capturing global galaxy structure


*This structure is intentionally modular and scalable.*

