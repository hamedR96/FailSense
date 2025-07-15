from inference import batch_inference
from load_dataset import load_data
from visualization import visualization_report
import torch

if __name__ == '__main__':

    model_ids = ["ACIDE/FailSense-AHA-Calvin-1p-3b",
                "ACIDE/FailSense-AHA-Calvin-2p-3b",
                "ACIDE/FailSense-Video-Calvin-1p-3b",
                "ACIDE/FailSense-Video-Calvin-2p-3b"
                ]

    dataset_names = ["droid",
                     "aha",
                     "calvin"]

    for model_id in model_ids:
        for dataset_name in dataset_names:
            torch.cuda.empty_cache()
            model_name=model_id.split("/")[-1]

            parts = model_name.split("-")
            style = "video" if "Video" in parts else "image"
            pov = 1 if "1p" in parts else 2

            test_sample = load_data(dataset_name=dataset_name, split="test", num_entry="full", style=style, pov=pov)

            predictions,labels=batch_inference(model_id=model_id,
                                        dataset=test_sample, batch_size=1,device="mps")

            visualization_report(labels, predictions, model_name=dataset_name+"-"+model_name)
