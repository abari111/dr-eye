""" All intermediate functions are packed here"""

import os
import random

import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def balance_data(
    df: DataFrame,
    n_classes: int = 5,
    n_samples: int = 200,
    save: bool = True,
    dir_path: str = None,
) -> DataFrame:
    """Balance the dataset with n_samples of each classes samples"""
    data_per_labels = {}
    for label in range(n_classes):
        data = df[df["diagnosis"] == label]
        data_per_labels[label] = data[:n_samples]

    # data_stats = {i: len(data) for i, data in data_per_labels.items()}
    balanced_data = pd.concat(data_per_labels.values(), ignore_index=True)

    if save:
        file_name = f"annot_balanced_{n_samples}.csv"
        if dir_path is None:
            dir_path = ""
        file_path = os.path.join(dir_path, file_name)

        balanced_data.to_csv(file_path, index=False)

    return balanced_data


class RetDataset(Dataset):
    """load retinopathy dataset"""

    def __init__(self, data_path: str, annot_path: str = None, transform=None) -> None:
        self.annots = pd.read_csv(annot_path)
        self.data_path = data_path
        self.transform = transform

    def __len__(
        self,
    ) -> int:
        return len(self.annots)

    def __getitem__(self, idx) -> torch.Tensor:
        filename, label = self.annots.iloc[idx]
        file_path = os.path.join(self.data_path, filename + ".png")
        img = Image.open(file_path)

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = ToTensor()(img)
            img_tensor = Resize((256, 256))(img_tensor)

        return img_tensor, torch.tensor(label)


class GenRetSample:
    """Generate randomly dataset sample: img, target"""

    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.random_idxs = random.sample(range(self.dataset_len), self.dataset_len)
        self.index = 0

    def __iter__(
        self,
    ):
        return self

    def __next__(self):
        if len(self.dataset) > self.index:
            self.index += 1
            return self.dataset[self.random_idxs[self.index]]
        else:
            raise StopIteration()


def display(dataset):
    gen = GenRetSample(dataset)
    img, lab = next(gen)

    img_np = img.numpy()
    img_np = img_np.swapaxes(0, 2)

    plt.imshow(img_np)
    plt.axis("off")
    plt.text(
        10,
        10,
        str(lab.item()),
        color="white",
        fontsize=12,
        bbox=dict(facecolor="black", alpha=0.5),
    )
    plt.show()


def display_gui(dataset):

    gen = GenRetSample(dataset)

    fig, ax = plt.subplots()

    def draw():
        img, lab = next(gen)

        img_np = img.numpy()
        img_np = img_np.swapaxes(0, 2)
        ax.imshow(img_np)
        ax.axis("off")
        canvas.draw()

    root = tk.Tk()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    button_quit = tk.Button(master=root, text="Next", command=draw)
    button_quit.pack(side=tk.BOTTOM)

    root.mainloop()


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    dataset_path = "datasets/ret_dataset/"
    annot_path = "datasets/annot_balanced_200.csv"
    n_classes = 5

    transform = Compose([ToTensor(), Resize((256, 256))])
    dataset = RetDataset(dataset_path, annot_path, transform)
    display_gui(dataset)
