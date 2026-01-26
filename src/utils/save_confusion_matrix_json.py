import json
from pathlib import Path
import torch
from sklearn.metrics import confusion_matrix

@torch.no_grad()
def save_confusion_matrix_json(
    model,
    dataloader,
    class_names,
    device,
    output_path,
    model_name,
    training_data,
    test_data
):
    model.eval()

    y_true = []
    y_pred = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    cm = confusion_matrix(y_true, y_pred)

    result = {
        "model": model_name,
        "training_data": training_data,
        "test_data": test_data,
        "num_classes": len(class_names),
        "class_names": class_names,
        "confusion_matrix": cm.tolist()
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved confusion matrix → {output_path}")
