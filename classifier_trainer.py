import torch
import torch.nn as nn
import tqdm
from load_dataset import load_data
from model import FailSense
from model import process_input

torch.manual_seed(42)


if __name__ == "__main__":
    vlm_model_id = "ACIDE/FailSense-Video-Calvin-1p-3b"  # Replace with your model ID

    model_name = vlm_model_id.split("/")[-1]
    parts = model_name.split("-")
    style = "video" if "Video" in parts else "image"
    pov = 1 if "1p" in parts else 2

    model = FailSense(vlm_model_id, device="mps")

    #model.load_classifier("/Users/hamed/PycharmProjects/FailSense/ACIDE/FailSense-AHA-Calvin-1p-3b/head_components.pt")
    train_dataset = load_data(dataset_name="calvin", style=style, split="train", pov=pov)
    #test_dataset = load_data(dataset_name="calvin", style="video", split="test", pov=1)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(list(model.classifier.parameters()) +
                                  list(model.attention_layer.parameters()), lr=1e-4, weight_decay=0.01)

    num_epochs = 1
    batch_size = 2

    best_val_acc = 0
    eval_step=100
    save_step=500

    start_batch_idx = 0  # <-- Change to your resume point

    total_batches = (len(train_dataset) + batch_size - 1) // batch_size

    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, i in enumerate(tqdm.tqdm(range(0, len(train_dataset), batch_size), desc="steps")):

            #if batch_idx < start_batch_idx:
                #continue  # Skip already processed batches

            batch_end = min(i + batch_size, len(train_dataset))
            entries= train_dataset[i:batch_end]

            try:
                batch_images = entries["images"]
                batch_texts = [process_input(entries["images"][z], entries["task"][z])for z in range(len(batch_images))]
                batch_labels = [0 if entry in ("0", "fail",0) else 1 for entry in entries["label"]]

                optimizer.zero_grad()
                logits = model(batch_images, batch_texts)

                batch_labels = torch.tensor(batch_labels,dtype=torch.float32, device=model.device)
                loss = criterion(logits.squeeze(), batch_labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                if (batch_idx + 1) % save_step == 0:
                    model.save_classifier(path="./"+vlm_model_id)

                # Statistics
                total_loss += loss.item()
                predictions = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
            except Exception as e:
                print(f"[Warning] Skipping batch {batch_idx} due to error: {e}")
                continue

        model.save_classifier(path="./"+vlm_model_id)
        print(f"Epoch Train Loss: {total_loss:.4f}, Epoch Train Acc: {correct / total:.4f}")
