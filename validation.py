import torch

from model import process_input


def validate(model, test_dataset, batch_size=2):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch_end = min(i + batch_size, len(test_dataset))

            entries = test_dataset[i:batch_end]

            batch_images = entries["images"]
            batch_texts = [process_input(entries["images"][i], entries["task"][i])for i in range(len(batch_images))]
            batch_labels = [0 if entry in ("0", "fail") else 1 for entry in entries["label"]]

            logits = model(batch_images, batch_texts)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32, device=model.device)
            predictions = (torch.sigmoid(logits.squeeze()) > 0.5).float()
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")
    return val_acc
