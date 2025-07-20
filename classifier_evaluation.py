from classifer_inference import classifier_batch_inference
from load_dataset import load_data
from model import FailSense
import torch
from visualization import visualization_report

import pickle

if __name__ == '__main__':

    batch_size=16

    model_ids = ["ACIDE/FailSense-AHA-Calvin-1p-3b",
                    #"ACIDE_1/FailSense-AHA-Calvin-2p-3b",
                   # "ACIDE_1/FailSense-Video-Calvin-1p-3b",
                    #"ACIDE_1/FailSense-Video-Calvin-2p-3b"
                    ]

    dataset_names = ["calvin",
                     "droid",
                    "aha",
                         ]

    for model_id in model_ids:

        model = FailSense(model_id, device="mps")
        model.load_classifier("/Users/hamed/PycharmProjects/FailSense/" + model_id + "/head_components.pt")

        for dataset_name in dataset_names:
            model_name = model_id.split("/")[-1]
            parts = model_name.split("-")
            style = "video" if "Video" in parts else "image"
            pov = 1 if "1p" in parts else 2

            test_sample = load_data(dataset_name=dataset_name, split="test", num_entry="full", style=style, pov=pov)

            predictions, labels = classifier_batch_inference(model, test_sample, batch_size=batch_size)

            #with open("results_"+dataset_name+"_"+model_name+ ".pkl", "wb") as f:
                #pickle.dump((predictions, labels), f)

           #with open("results_"+dataset_name+"_"+model_name+ ".pkl", "rb") as f:
                #predictions, labels = pickle.load(f)

            #print(predictions)
            #print(labels)

            visualization_report(labels, predictions, model_name=dataset_name + "-" + model_name,output_dir="./classifier_results3")


        #torch.mps.empty_cache()