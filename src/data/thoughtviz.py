import cv2
import torch as pt
from torch.utils.data import Dataset
from pathlib import Path


class ThoughtVizDepthBasic(Dataset):
    def __init__(
        self,
        data_pth: Path | str,
        imagenet_dir: Path | str,
        ext: str = "JPEG",
    ):
        self.data_pth = Path(data_pth)
        self.imagenet_dir = Path(imagenet_dir)
        self.ext = ext
        self.data = pt.load(data_pth, weights_only=False)
        self.cein = None
        x = self.data["dataset"][0]["eeg"]
        row = pt.arange(x.shape[0]).view(-1, 1).repeat(1, x.shape[0]).view(-1)
        col = pt.arange(x.shape[0]).view(-1, 1).repeat(x.shape[0], 1).view(-1)
        self.cein = pt.stack([row, col], dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg = self.data["dataset"][idx]["eeg"]
        eeg = eeg[:, 20:470]
        grp = (eeg, self.cein)
        img_idx = self.data["dataset"][idx]["image"]
        img_name = self.data["images"][img_idx]
        cls_name = img_name.split("_")[0]
        img_path = self.imagenet_dir / cls_name / f"{img_name}.{self.ext}"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        return grp, img
