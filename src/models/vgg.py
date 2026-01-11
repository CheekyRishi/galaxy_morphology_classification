from torch import nn
import torchvision
import torch

def vgg16():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weights = torchvision.models.VGG16_Weights.DEFAULT
    model = torchvision.models.vgg16(weights=weights).to(device)
 
    for param in model.features.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.2),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.2),
        nn.Linear(4096, 10) 
    ).to(device)

    return model

def vgg19():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weights = torchvision.models.VGG19_Weights.DEFAULT
    model = torchvision.models.vgg19(weights=weights).to(device)
    
    for param in model.features.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10) 
    ).to(device)

    return model