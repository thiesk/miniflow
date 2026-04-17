from sklearn.datasets import make_moons
from torch.utils.data import Dataset
import torch


class MoonDataset(Dataset):
    def __init__(self, config):
        self.config = config
        n_samples = self.config["data"]["n_samples"]
        seed = self.config["data"]["seed"]
        noise = self.config["data"]["noise"]
        data, _ = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        self.data = torch.Tensor(data).to(torch.float)

    def __len__(self):
        return self.config["data"]["n_samples"]

    def __getitem__(self, idx):

        coords = self.data[idx]

        return coords



