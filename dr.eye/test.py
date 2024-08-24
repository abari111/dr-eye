import torch

from utils import RetDataset, GenRetSample
from models import get_mobilenet_v3


@torch.no_grad
def test(model, dataloader_test):
    model.eval()
    acc = 0
    data_size = len(dataloader_test.dataset)
    for x, y in dataloader_test:
        y_pred = model(x)
        acc += (y_pred.argmax(dim=1) == y).sum()
    print(f"test accuracy: {acc/data_size}")


if __name__ == "__main__":

    dataset_path = "datasets/ret_dataset/"
    annot_path = "datasets/annot_balanced_200.csv"
    dataset = RetDataset(dataset_path, annot_path)

    gen = GenRetSample(dataset)
    img, lab = next(gen)

    model = get_mobilenet_v3()
    checkpoint = torch.load("models/checkpoint_0.pth")
    model.load_state_dict(checkpoint["model_st_dict"])
    print(torch.argmax(model(img.unsqueeze(dim=0))).item(), lab.item())
