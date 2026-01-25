import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image

from config import DEVICE, VIT_IMAGE_SIZE,VIT_PATCH_SIZE, IMAGE_SIZE, CLASS_NAMES, CHECKPOINTS
from src.data.transforms import (
    get_eval_transforms,
    get_vit_eval_transform
)

from src.models.get_model import get_model

_MODEL_CACHE = {}

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
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    checkpoint_path = CHECKPOINTS[model_name]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    model_or_tuple = get_model(model_name, num_classes=len(CLASS_NAMES))


    # Handle CNNs returning (model, transform)
    if isinstance(model_or_tuple, tuple):
        model = model_or_tuple[0]
    else:
        model = model_or_tuple

    state = torch.load(checkpoint_path, map_location=DEVICE)
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
def predict(image: Image.Image, model_name: str):
    if image is None:
        return "No image provided", {}

    model = load_model(model_name)
    transform = get_transform(model_name)

    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    if model_name.lower() == "vit":
        tensor = patchify(tensor, patch_size=VIT_PATCH_SIZE)

    logits = model(tensor)

    probs = F.softmax(logits, dim=1).squeeze(0)

    probs_dict = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    top_class = max(probs_dict, key=probs_dict.get)
    confidence = probs_dict[top_class]

    return f"{top_class} ({confidence:.3f})", probs_dict


# ---- GRADIO UI ----
with gr.Blocks(title="Galaxy Morphology Classification") as demo:
    gr.Markdown("## Galaxy Morphology Classification")
    gr.Markdown("Upload a galaxy image and select a trained model")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Galaxy Image")
        model_dropdown = gr.Dropdown(
            choices=list(CHECKPOINTS.keys()),
            value="resnet50",
            label="Model"
        )

    predict_btn = gr.Button("Predict")

    label_output = gr.Textbox(label="Prediction")
    prob_output = gr.Label(label="Class Probabilities")

    predict_btn.click(
        fn=predict,
        inputs=[image_input, model_dropdown],
        outputs=[label_output, prob_output]
    )


if __name__ == "__main__":
    demo.launch(share=True)
