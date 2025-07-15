from load_dataset import load_data

# Example usage
if __name__ == "__main__":

    # Load full training dataset with default parameters (image, POV 1)
    calvin_train = load_data(dataset_name="calvin", split="train", num_entry="full")
    print(f"Calvin train dataset size: {len(calvin_train)}")
    print("###############################")
    # Load random sample of test dataset with video type and POV 2
    calvin_test_sample =load_data(dataset_name="calvin", split="test", num_entry=100, style="video", pov=2)
    print(f"Calvin test sample size: {len(calvin_test_sample)}")
    print("###############################")
    # Load droid test dataset with image type and POV 1
    droid_test = load_data(dataset_name="droid", split="test", style="image", pov=1)
    print(f"Droid test dataset size: {len(droid_test)}")
    print("###############################")
    # Load AHA training dataset with video type and POV 1
    aha_train = load_data(dataset_name="aha",split= "train", num_entry="full", style= "video", pov=1)
    print(f"AHA train dataset size: {len(aha_train)}")
