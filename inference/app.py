import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download

from config import DEVICE, VIT_IMAGE_SIZE,VIT_PATCH_SIZE, IMAGE_SIZE, CLASS_NAMES, CHECKPOINTS
from src.data.transforms import (
    get_eval_transforms,
    get_vit_eval_transform
)

from src.models.get_model import get_model

_MODEL_CACHE = {}

MODEL_LABELS = {
    "Vision Transformer (ViT)": "vit",
    "ResNet 18": "resnet18",
    "ResNet 26": "resnet26",
    "ResNet 50": "resnet50",
    "Custom CNN": "custom_cnn",
    "VGG 16": "vgg16",
    "VGG 19": "vgg19",
}


def patchify(image_tensor, patch_size):
    """
    image_tensor: (1, 3, H, W)
    returns: (1, num_patches, patch_dim)
    """
    _, c, h, w = image_tensor.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
        f"Image size {h}x{w} not divisible by patch size {patch_size}")

    patches = image_tensor.unfold(2, patch_size, patch_size) \
                           .unfold(3, patch_size, patch_size)

    patches = patches.contiguous().view(
        image_tensor.size(0),
        c,
        -1,
        patch_size,
        patch_size
    )

    patches = patches.permute(0, 2, 1, 3, 4)
    patches = patches.flatten(2)

    return patches

def load_model(model_name: str):
    model_name = model_name.lower()

    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    ckpt_path = hf_hub_download(
        repo_id="Rishabh-9090/galaxy-morphology-models",
        filename=CHECKPOINTS[model_name],
        repo_type="model"
    )

    model = get_model(model_name, num_classes=len(CLASS_NAMES))

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    _MODEL_CACHE[model_name] = model
    return model



def get_transform(model_name: str):
    name = model_name.lower()

    if name == "vit":
        return get_vit_eval_transform(VIT_IMAGE_SIZE)

    return get_eval_transforms(IMAGE_SIZE)

# ---- INFERENCE ----
@torch.inference_mode()
def predict(image: Image.Image, model_label: str):
    if image is None:
        return "No image provided", {}

    # Map UI label -> internal model key
    model_name = MODEL_LABELS[model_label]

    model = load_model(model_name)
    transform = get_transform(model_name)

    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    if model_name.lower() == "vit":
        tensor = patchify(tensor, patch_size=VIT_PATCH_SIZE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    # Full probability dictionary (useful for plots / tables)
    probs_dict = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    # Top 3 predictions
    topk = torch.topk(probs, k=3)

    results = []
    for idx, score in zip(topk.indices, topk.values):
        class_name = CLASS_NAMES[idx.item()]
        results.append(f"{class_name}: {score.item():.3f}")

    top3_text = "\n".join(results)

    return top3_text, probs_dict



# ---- GRADIO UI ----
with gr.Blocks(title="Galaxy Morphology Classification") as demo:
    gr.Markdown("## Galaxy Morphology Classification")
    gr.Markdown("Upload a galaxy image and select a trained model")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Galaxy Image")
        model_dropdown = gr.Dropdown(
        choices=list(MODEL_LABELS.keys()),
        value="Vision Transformer (ViT)",
        label="Model"
    )
        
    predict_btn = gr.Button("Predict")

    label_output = gr.Textbox(label="Top 3 Prediction")
    prob_output = gr.Label(label="Class Probabilities")

    predict_btn.click(
        fn=predict,
        inputs=[image_input, model_dropdown],
        outputs=[label_output, prob_output]
    )


if __name__ == "__main__":
    demo.launch(share=True)
