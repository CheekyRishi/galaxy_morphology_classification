from pathlib import Path
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
VIT_IMAGE_SIZE = 200
VIT_PATCH_SIZE = 25

CLASS_NAMES = [
    "barred_spiral",
    "cigar_shaped",
    "disturbed",
    "edge_on_no_bulge",
    "edge_on_with_bulge",
    "in_between_round_smooth",
    "merging",
    "round_smooth",
    "unbarred_loose_spiral",
    "unbarred_tight_spiral",
]

CHECKPOINTS = {
    "resnet18": "resnet18/Resnet18_29_epochs_trainable_classifier_early_stopping.pth",
    "resnet26": "resnet26/Resnet26_33_epochs_trainable_classifier_early_stopping.pth",
    "resnet50": "resnet50/Resnet50_62_epochs_trainable_classifier_early_stopper.pth",
    "custom_cnn": "custom_cnn/CustomCNN_25_epochs_early_stopping.pth",
    "vgg16": "vgg16/VGG16_15_epochs_trainable_classifier_early_stopping.pth",
    "vgg19": "vgg19/VGG19_60_epochs_trainable_classifier_early_stopping.pth",
    "vit": "vit/vit_best_balanced_dataset.pth",
}

