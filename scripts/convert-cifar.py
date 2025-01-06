"""
You need to download CIFAR10 data from https://www.cs.toronto.edu/~kriz/cifar.html
Then using this script to extract data to image
Script copied from: https://gist.github.com/fzliu/64821d31816bce595a4bbd98588b37f5
"""

import os
import pickle
from pathlib import Path

import numpy as np
import tqdm
from skimage.io import imsave

CIFAR_DIR = "data/cifar-10"


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def save_as_image(img_flat, fname, train=True):
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    if train:
        imsave(os.path.join(f"{CIFAR_DIR}/train/", fname), img)
    else:
        imsave(os.path.join(f"{CIFAR_DIR}/test/", fname), img)


if __name__ == "__main__":
    # make directory if not exists
    Path("data/cifar-10").mkdir(parents=True, exist_ok=True)
    Path("data/cifar-10/train").mkdir(parents=True, exist_ok=True)
    Path("data/cifar-10/test").mkdir(parents=True, exist_ok=True)

    # Load and write image
    labels = dict()
    label_names = dict()
    source_dir = Path("data/cifar-10-batches-py/")
    files = source_dir.iterdir()
    for file in files:
        if file.suffix == ".html":
            continue
        elif file.suffix == ".meta":
            data = unpickle(file)
            label_names = {
                idx: name.decode("utf-8")
                for idx, name in enumerate(data[b"label_names"])
            }
        else:
            data = unpickle(file)
            print(f"Write image for {str(file)}")
            for i in tqdm.trange(10000):
                img_flat = data[b"data"][i]
                fname = data[b"filenames"][i].decode()
                label = data[b"labels"][i]
                if file.name == "test_batch":
                    save_as_image(img_flat, fname, False)
                else:
                    save_as_image(img_flat, fname, True)
                labels[fname] = label

    # Write out labels file
    with open(f"{CIFAR_DIR}/labels_img.csv", "w+") as f:
        for fname, label in labels.items():
            f.write("{0},{1}\n".format(fname, label))

    # Write label type
    with open(f"{CIFAR_DIR}/labels_name.csv", "w+") as f:
        for idx, name in label_names.items():
            f.write("{0},{1}\n".format(idx, name))
