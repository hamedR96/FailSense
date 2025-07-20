import torch
import tqdm

from FailSense.inference import process_input


def classifier_batch_inference(model, dataset, batch_size):
    predictions = []
    labels = []

    total_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"Processing {total_batches} batches...")

    # Process dataset in batches
    for batch_idx, i in enumerate(tqdm.tqdm(range(0, len(dataset), batch_size), desc="Processing batches")):
        batch_end = min(i + batch_size, len(dataset))

        # Prepare batch data
        batch_images = []
        batch_texts = []
        batch_labels = []

        for j in range(i, batch_end):
            entry = dataset[j]
            images = entry["images"]
            text = entry["task"]
            label = 0 if entry["label"] == "0" or entry["label"] == "fail" else 1

            prompt = process_input(images, text)

            batch_images.append(images)
            batch_texts.append(prompt)
            batch_labels.append(label)

        batch_preds, _ = model.predict(batch_images, batch_texts)

        # Ensure batch_preds is a list of scalars, not a single scalar
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.view(-1).cpu().tolist()
        elif isinstance(batch_preds, int) or isinstance(batch_preds, float):
            batch_preds = [batch_preds]

        predictions.extend(batch_preds)

        labels.extend(batch_labels)

    print("-" * 50)
    print(f"Inference completed!")
    print(f"Total predictions: {len(predictions)}")
    print(f"Total labels: {len(labels)}")

    # Print summary
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)

    accuracy = correct / len(predictions) * 100

    print(f"\nFinal Summary:")
    print(f"Total items: {len(predictions)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    return predictions, labels