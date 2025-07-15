import torch
import tqdm
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import os
from datetime import datetime
from peft import PeftModel

def process_input(images, text):
    if not isinstance(images, list):
        prompt = "<image> evaluate en " + text
    else:
        prompt = "<image>" * len(images) + " evaluate en " + text
    return prompt


def batch_infer(model, processor, batch_inputs, batch_input_lens):
    with torch.inference_mode():
        # Generate for the entire batch
        generations = model.generate(**batch_inputs, max_new_tokens=100, do_sample=False)

        # Decode each generation in the batch, removing input tokens
        batch_predictions = []
        for i, generation in enumerate(generations):
            # Remove input tokens for each item in batch
            generation_only = generation[batch_input_lens[i]:]
            pred = processor.decode(generation_only, skip_special_tokens=True)
            batch_predictions.append(pred)

    return batch_predictions


def batch_inference(model_id, dataset, batch_size=4, device="cuda", output_dir="./results"):
    predictions = []
    labels = []

    device = torch.device(device)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting batch inference...")
    print(f"Model: {model_id}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")
    print("✓ Processor loaded")

    base_model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-mix-224",
                                                                   device_map=device)
    model = PeftModel.from_pretrained(base_model, model_id).eval()

    print("✓ Model loaded and moved to device")
    print("-" * 50)

    total_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"Processing {total_batches} batches...")

    # Process dataset in batches
    for batch_idx, i in enumerate(tqdm.tqdm(range(0, len(dataset), batch_size), desc="Processing batches")):
        batch_end = min(i + batch_size, len(dataset))

        print(
            f"Batch {batch_idx + 1}/{total_batches}: Processing items {i + 1}-{batch_end}")

        # Prepare batch data
        batch_images = []
        batch_texts = []
        batch_labels = []

        for j in range(i,batch_end):
            entry=dataset[j]
            images = entry["images"]
            text = entry["task"]
            label = "fail" if entry["label"] == "0" or entry["label"] == "fail" else "success"

            prompt = process_input(images, text)

            batch_images.append(images)
            batch_texts.append(prompt)
            batch_labels.append(label)

        # Process entire batch at once
        batch_inputs = processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True
        ).to(device)

        # Get input lengths for each item in batch (for removing from generation)
        batch_input_lens = [len(input_ids) for input_ids in batch_inputs["input_ids"]]

        # Run inference on the batch
        batch_preds = batch_infer(model, processor, batch_inputs, batch_input_lens)

        # Collect results
        predictions.extend(batch_preds)
        labels.extend(batch_labels)

        print(f"  ✓ Batch {batch_idx + 1} completed")

    print("-" * 50)
    print(f"Inference completed!")
    print(f"Total predictions: {len(predictions)}")
    print(f"Total labels: {len(labels)}")

    # Save results to file
    model_name=model_id.split("/")[-1]
    output_file = os.path.join(output_dir, f"{model_name}.txt")

    print(f"Saving results to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Batch Inference Results\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Dataset size: {len(dataset)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 50 + "\n\n")

        f.write("Results:\n")
        f.write("-" * 30 + "\n")

        for i, (pred, label) in enumerate(zip(predictions, labels)):
            f.write(f"Item {i + 1}:\n")
            f.write(f"  Label: {label}\n")
            f.write(f"  Prediction: {pred}\n")
            f.write(f"  Match: {'✓' if pred.strip().lower() == label.lower() else '✗'}\n")
            f.write("-" * 30 + "\n")

        # Calculate accuracy
        correct = sum(1 for pred, label in zip(predictions, labels)
                      if pred.strip().lower() == label.lower())
        accuracy = correct / len(predictions) * 100

        f.write(f"\nSummary:\n")
        f.write(f"Total items: {len(predictions)}\n")
        f.write(f"Correct predictions: {correct}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")

    print(f"✓ Results saved to {output_file}")

    # Print summary
    correct = sum(1 for pred, label in zip(predictions, labels)
                  if pred.strip().lower() == label.lower())
    accuracy = correct / len(predictions) * 100

    print(f"\nFinal Summary:")
    print(f"Total items: {len(predictions)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    return predictions, labels