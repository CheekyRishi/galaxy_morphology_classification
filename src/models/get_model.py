import torch.nn as nn
from torchvision import models

from src.models.resnet import resnet50, resnet26, resnet18
from src.models.vit import VisionTransformer
from src.models.vgg import vgg16, vgg19
from src.models.custom_cnn import custom_cnn

def get_model(name: str, num_classes: int):
    name = name.lower()

    if name == "resnet18":
        model = resnet18(num_classes=num_classes)

    elif name == "resnet26":
        model = resnet26(num_classes=num_classes)

    elif name == "resnet50":
        model = resnet50(num_classes=num_classes)

    elif name == "vgg16":
        model = vgg16(num_classes=num_classes)

    elif name == "vgg19":
        model = vgg19(num_classes=num_classes)

    elif name == "custom_cnn":
        model = custom_cnn(num_classes=num_classes)
    elif name == "vit":
        config = {
            "patch_size": 25,
            "num_channels": 3,
            "num_patches": 64,          
            "hidden_dim": 256,
            "num_heads": 8,
            "mlp_dim": 1024,
            "num_layers": 6,
            "dropout_rate": 0.1,
            "num_classes": 10,
        }
        model = VisionTransformer(config)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model
