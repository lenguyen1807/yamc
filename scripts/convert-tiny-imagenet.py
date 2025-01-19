import os
from pathlib import Path

import pandas as pd
import tqdm

IMAGE_NET_DIR = "data/tiny-imagenet-200"

if __name__ == "__main__":
    label_ids = dict()
    idx = 0
    for f in os.scandir(f"{IMAGE_NET_DIR}/train"):
        if f.is_dir():
            label_ids[f.name] = idx
            idx += 1

    # Load test data label
    test_label_data = pd.read_csv(
        f"{IMAGE_NET_DIR}/val/val_annotations.txt",
        sep="\t",
        names=["id", "label", "0", "1", "2", "3"],
    )
    test_label_data.drop(["0", "1", "2", "3"], axis=1, inplace=True)
    test_label_data.set_index("id", inplace=True)
    test_label_data["label"] = test_label_data["label"].apply(lambda x: label_ids[x])

    # Now write test label to file
    test_label_data.to_csv(f"{IMAGE_NET_DIR}/val_labels.csv", header=False)

    # Write label map to file
    label_ids_pd = pd.DataFrame(label_ids.items(), columns=["name", "label"])
    label_ids_pd.to_csv(f"{IMAGE_NET_DIR}/label_ids.csv", header=False, index=False)
