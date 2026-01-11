import torch
from torch import nn
import torchvision
from torch import nn
import timm

def resnet50(num_classes=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load weights and model
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).to(device)
    automatic_transform = weights.transforms()
    # 2. Freeze all base layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    updated_classifier = nn.Sequential(
        nn.Dropout(inplace=True,p=0.2),
        nn.Linear(in_features=1280,out_features=num_classes,bias=True)
    ).to(device)

    model.fc = updated_classifier

    return model,weights.transforms()


def resnet26(num_classes=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = timm.create_model('resnet26', pretrained=True).to(device)
    
    for param in model.parameters():
        param.requires_grad = False
        
    in_features = model.get_classifier().in_features
    
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    ).to(device)

    # Get the specific transforms required for this model
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    return model, transforms

def resnet18(num_classes=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights).to(device)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.fc = nn.Linear(512, num_classes).to(device)
    
    return model,weights.transforms()